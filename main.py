#!/usr/bin/env python

import os.path
from math import ceil
from xml.dom import minidom
from collections import namedtuple
from subprocess import check_call
from tempfile import NamedTemporaryFile
import cPickle as pickle
from functools import partial

import requests

from seaborn import xkcd_rgb as COLORS
SILVER = COLORS['silver']
GREEN = COLORS['medium green']

from matplotlib import pyplot as plt

from scipy.signal import decimate

# sudo apt-get install libsndfile-dev libasound2-dev 
# pip install scikits.audiolab
import scikits.audiolab as audio 


class Sound(object):
    signal = []
    rate = None
    bits = None

    def __init__(self, signal, rate, bits):
        self.signal = signal
        self.rate = rate
        self.bits = bits
    
    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start
            if start:
                start = int(start * self.rate)
            stop = item.stop
            if stop:
                stop = int(stop * self.rate)
            return Sound(self.signal[start:stop], self.rate, self.bits)

    def decimate(self, rate):
        factor = self.rate / rate
        signal = decimate(self.signal, factor)
        return Sound(signal, rate, self.bits)

    @property
    def seconds(self):
        return float(len(self.signal)) / self.rate

    def __repr__(self):
        return ('Sound(signal={0.signal!r}, '
                'rate={0.rate!r}, bits={0.bits!r})'
                .format(self))


MALE = 'M'
FEMALE = 'F'
Segment = namedtuple('Segment', ['start', 'stop', 'speaker', 'gender'])


class Option(namedtuple('Transcript', ['text', 'confidence'])):
    def _repr_pretty_(self, printer, _):
        printer.begin_group(7, 'Option(')
        printer.text(u'text="{}"'.format(self.text))
        printer.text(',')
        printer.breakable()
        printer.text('confidence={}'.format(self.confidence))
        printer.end_group(7, ')')


def read_sounds(dir='sounds', names=['mosenergo', 'rigla', 'transaero', 'water', 'worldclass', 'test']):
    sounds = {}
    for name in names:
        path = os.path.join(dir, name + '.wav')
        sounds[name] = read_sound(path)
    return sounds


def read_sound(path):
    signal, rate, bits = audio.wavread(path)
    return Sound(signal, rate, bits)


def write_sound(sound, path):
    audio.wavwrite(sound.signal, path, fs=sound.rate, enc=sound.bits)


def plot_sound(sound, sample=10, linewidth=0.1, height=1.5, step=20):
    sound = sound.decimate(sound.rate / sample)
    rate = sound.rate
    length = len(sound.signal)
    seconds = float(length) / rate
    rows = int(ceil(seconds / step))
    fig = plt.figure(figsize=(step, height * rows))
    width = step * rate
    positions = range(0, width, rate)
    axes = []
    for row in xrange(rows):
        start = row * step
        stop = (row + 1) * step
        slice = sound[start:stop]
        labels = [_ for _ in xrange(start, stop) if _ < seconds]
        axis = fig.add_subplot(rows, 1, row + 1)
        axis.get_yaxis().set_ticks([])
        axis.grid(False)
        axis.plot(slice.signal, linewidth=linewidth)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_ylim(-1, 1)
        axis.set_xlim(0, width)
        axes.append(axis)
    return axes


def read_diarization(path):
    xml = minidom.parse(path)
    genders = {}
    for speaker in xml.getElementsByTagName('speaker'):
        attributes = speaker.attributes
        name = attributes['name'].value
        gender = attributes['gender'].value
        genders[name] = gender
    for segment in xml.getElementsByTagName('segment'):
        attributes = segment.attributes
        start = float(attributes['start'].value)
        stop = float(attributes['end'].value)
        speaker = attributes['speaker'].value
        gender = genders[speaker]
        yield Segment(start, stop, speaker, gender)


def diarize_sound(sound, lium='lium_spkdiarization-8.4.1.jar'):
    with NamedTemporaryFile(suffix='.wav') as tmp:
        write_sound(sound, tmp.name)
        with NamedTemporaryFile(suffix='.xml') as dump:
            check_call(['java', '-jar', lium, '--fInputMask', tmp.name,
                        '--sOutputFormat', 'seg.xml,UTF8',
                        '--sOutputMask', dump.name,
                        '--doCEClustering', 'sound'])
            return list(read_diarization(dump.name))


def load_diarization(path):
    with open(path) as dump:
        return [Segment(*_) for _ in pickle.load(dump)]


def dump_diarization(segments, path):
    with open(path, 'w') as dump:
        pickle.dump([tuple(_) for _ in segments], dump)


def plot_segment(start, stop, width, speaker, axis,
                 shift=-0.75, color=SILVER):
    if speaker:
        axis.text(start, shift - 0.1, speaker)
    axis.axhline(
        shift, start / width + 0.01, stop / width,
        color=color, alpha=0.5
    )


def plot_segments(sound, guess, etalon, sample=10, linewidth=0.1, height=1.5, step=20):
    axes = plot_sound(
        sound, sample=sample, linewidth=linewidth,
        height=height, step=step
    )
    rate = sound.rate / sample
    width = step * rate
    for segments, plot in [
        (guess, partial(plot_segment, shift=-0.75, color=SILVER)),
        (etalon, partial(plot_segment, shift=-0.5, color=GREEN))
    ]:
        for segment in segments:
            speaker = segment.speaker
            start = segment.start * rate
            stop = segment.stop * rate
            start_row = int(start / width)
            stop_row = int(stop / width)
            if start_row == stop_row:
                row = start_row
                zero = row * width
                start -= zero
                stop -= zero
                axis = axes[row]
                plot(start, stop, width, speaker, axis)
            else:
                axis = axes[start_row]
                start = start - start_row * width
                plot(start, width, width, speaker, axis)
                for row in xrange(start_row + 1, stop_row):
                    axis = axes[row]
                    plot(0, width, width, None, axis)
                axis = axes[stop_row]
                stop = stop - stop_row * width
                plot(0, stop, width, None, axis)
    return axes


def read_transcript(transcript):
    xml = minidom.parseString(transcript)
    transcript = []
    for option in xml.getElementsByTagName('variant'):
        confidence = float(option.attributes['confidence'].value)
        text = option.childNodes[0].data
        transcript.append(Option(text, confidence))
    return transcript


def transcribe_sound(sound):
    with NamedTemporaryFile(suffix='.wav') as tmp:
        path = tmp.name
        write_sound(sound, path)
        with open(path, 'rb') as dump:
            responce = requests.post(
                'https://asr.yandex.net/asr_xml',
                params={
                    'key': '3bf3afd8-10ba-46d6-9aef-712da393dc14',
                    'uuid': '32144111815349669875228586783736',
                    'topic': 'notes',
                    'lang': 'ru-RU'
                },
                headers={
                    'Content-Type': 'audio/x-wav'
                },
                data=dump
            )
            return read_transcript(responce.content)


def transcribe_segments(sound, segments):
    for segment in segments:
        slice = sound[segment.start:segment.stop]
        try:
            transcript = transcribe_sound(slice)
        except:
            transcript = None
        yield segment, transcript


def load_segments_transcript(path):
    transcript = []
    with open(path) as dump:
        for segment, part in pickle.load(dump):
            segment = Segment(*segment)
            if part:
                part = [Option(*_) for _ in part]
            transcript.append((segment, part))
        return transcript


def dump_segments_transcript(transcript, path):
    payload = []
    for segment, part in transcript:
        segment = tuple(segment)
        if part:
            part = [tuple(_) for _ in part]
        payload.append((segment, part))
    with open(path, 'w') as dump:
        pickle.dump(payload, dump)
