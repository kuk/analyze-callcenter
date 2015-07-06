#!/usr/bin/env python
# encoding: utf8

import os.path
from shutil import rmtree
from math import ceil
from xml.dom import minidom, getDOMImplementation
from collections import namedtuple
from subprocess import check_call
from tempfile import mkdtemp, NamedTemporaryFile
import cPickle as pickle
from collections import Counter
from functools import partial

import requests

from seaborn import xkcd_rgb as COLORS
SILVER = COLORS['silver']
BLUE = COLORS['denim blue']
GREEN = COLORS['medium green']
RED = COLORS['reddish orange']

from matplotlib import pyplot as plt

from scipy.signal import decimate

# sudo apt-get install libsndfile-dev libasound2-dev 
# pip install scikits.audiolab
import scikits.audiolab as audio 

from nltk.stem import SnowballStemmer


NAMES = ['mosenergo', 'rigla', 'transaero', 'water', 'worldclass']


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


def read_sounds(dir='sounds', names=NAMES):
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


def write_diarization(segments, path):
    impl = getDOMImplementation()
    epac = impl.createDocument(None, 'epac', None)
    audio = epac.documentElement.appendChild(epac.createElement('audiofile'))
    audio.setAttribute('name', 'sound')
    node = audio.appendChild(epac.createElement('speakers'))
    speakers = {(_.speaker, _.gender) for _ in segments}
    for name, gender in speakers:
        node = node.appendChild(epac.createElement('speaker'))
        node.setAttribute('name', name)
        node.setAttribute('gender', gender)
        node.setAttribute('generator', 'auto')
        node.setAttribute('identity', '')
        node.setAttribute('type', 'generic label')
    node = audio.appendChild(epac.createElement('segments'))
    for segment in segments:
        node = node.appendChild(epac.createElement('segment'))
        node.setAttribute('start', str(segment.start))
        node.setAttribute('end', str(segment.stop))
        node.setAttribute('speaker', segment.speaker)
        node.setAttribute('generator', 'auto')
        node.setAttribute('bandwidth', 'U')
    with open(path, 'w') as dump:
        dump.write(epac.toxml())


def diarize_sound(sound, lium='lium'):
    # Since LIUM was trained with 16khz data
    decimate = sound.decimate(16000)
    with NamedTemporaryFile(suffix='.wav') as file:
        path = file.name
        write_sound(decimate, path)
        data = mkdtemp()
        try:
            check_call(['./diarization.sh', path, data], cwd=lium)
            path = os.path.join(data, 'segments.seg')
            segments = read_diarization(path)
            segments = remove_silent_segment(segments, sound)
            segments = rename_segments(segments)
            segments = join_continuous_segments(segments)
            return segments
        finally:
            rmtree(data)


def remove_silent_segment(segments, sound):
    clean = []
    for segment in segments:
        slice = sound[segment.start:segment.stop]
        energy = (slice.signal ** 2).sum()
        density = energy / slice.seconds
        if density > 15:
            clean.append(segment)
    return clean


def rename_segments(segments):
    seconds = Counter()
    for segment in segments:
        seconds[segment.speaker] += (segment.stop - segment.start)
    mapping = {}
    top = [speaker for speaker, _ in seconds.most_common()]
    mapping[top[0]] = 'S0'
    for speaker in top[1:]:
        mapping[speaker] = 'S1'
    return [Segment(_.start, _.stop, mapping[_.speaker], _.gender) for _ in segments]


def join_continuous_segments(segments):
    join = []
    previous = None
    for segment in segments:
        if not previous:
            previous = segment
        elif previous.speaker == segment.speaker and previous.stop == segment.start:
            previous = Segment(
                previous.start, segment.stop,
                previous.speaker, previous.gender
            )
        else:
            join.append(previous)
            previous = segment
    join.append(previous)
    return join


def diff_segments(guess, etalon):
    etalon_start = 0
    etalon_stop = 1
    guess_start = 2
    guess_stop = 3
    points = []
    for index, segment in enumerate(etalon):
        points.append((segment.start, index, etalon_start, segment.speaker))
        points.append((segment.stop, index, etalon_stop, None))
    for index, segment in enumerate(guess):
        points.append((segment.start, index, guess_start, segment.speaker))
        points.append((segment.stop, index, guess_stop, None))
    points = sorted(points)
    diff = []
    no_etalon_no_guess = 0
    no_etalon_guess = 1
    etalon_no_guess = 2
    etalon_guess = 3
    state = no_etalon_no_guess
    previous = 0
    etalon_speaker = None
    guess_speaker = None
    for point, _, type, speaker in points:
        if previous != point:
            diff.append((previous, point, state, etalon_speaker, guess_speaker))
        previous = point
        if state == no_etalon_no_guess:
            if type == etalon_start:
                state = etalon_no_guess
                etalon_speaker = speaker
            elif type == guess_start:
                state = no_etalon_guess
                guess_speaker = speaker
        elif state == no_etalon_guess:
            if type == etalon_start:
                state = etalon_guess
                etalon_speaker = speaker
            elif type == guess_stop:
                state = no_etalon_no_guess
                guess_speaker = None
        elif state == etalon_no_guess:
            if type == etalon_stop:
                state = no_etalon_no_guess
                etalon_speaker = None
            elif type == guess_start:
                state = etalon_guess
                guess_speaker = speaker
        elif state == etalon_guess:
            if type == etalon_stop:
                state = no_etalon_guess
                etalon_speaker = None
            elif type == guess_stop:
                state = etalon_no_guess
                guess_speaker = None
    diff = [Segment(start, stop, None, None)
            for start, stop, state, etalon_speaker, guess_speaker in diff
            if (state == etalon_no_guess
                or (state == etalon_guess and etalon_speaker != guess_speaker))]
    return join_continuous_segments(diff)
            

def load_diarization(path):
    with open(path) as dump:
        return [Segment(*_) for _ in pickle.load(dump)]


def dump_diarization(segments, path):
    with open(path, 'w') as dump:
        pickle.dump([tuple(_) for _ in segments], dump)


def load_diarizations(dir='segments', format='{}.pickle', names=NAMES):
    diarizations = {}
    for name in names:
        path = os.path.join(dir, format.format(name))
        diarizations[name] = load_diarization(path)
    return diarizations


def plot_segment(start, stop, width, speaker, axis,
                 shift=-0.75, color=SILVER):
    if speaker is not None:
        axis.text(start, shift - 0.1, speaker)
    axis.axhline(
        shift, start / width + 0.01, stop / width,
        color=color, alpha=0.5
    )


def plot_segments(sound, guess=(), etalon=(), diff=True, sample=10, linewidth=0.1, height=1.5, step=20):
    axes = plot_sound(
        sound, sample=sample, linewidth=linewidth,
        height=height, step=step
    )
    rate = sound.rate / sample
    width = step * rate
    if diff:
        diff = diff_segments(guess, etalon)
    for segments, plot in [
        (guess, partial(plot_segment, shift=-0.45, color=BLUE)),
        (etalon, partial(plot_segment, shift=-0.7, color=GREEN)),
        (diff, partial(plot_segment, shift=-0.95, color=RED))
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
        # 48khz (default) sound accupies 0.1Mb/s so 10 seconds piece
        # as larger that 1Mb so I resample it to 22khz. It accupies
        # 0.05Mb/s so 20 seconds is more that 1Mb but lickely there
        # were no such segments
        if slice.seconds > 10:
            slice = slice.decimate(22000)
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
            part = [(_.text.decode('utf8'), _.confidence) for _ in part]
        payload.append((segment, part))
    with open(path, 'w') as dump:
        pickle.dump(payload, dump)


def load_transcripts(dir='transcripts', format='{}.pickle', names=NAMES):
    transcripts = {}
    for name in names:
        path = os.path.join(dir, format.format(name))
        transcripts[name] = load_segments_transcript(path)
    return transcripts


class Const(namedtuple('Const', 'value')):
    def dump(self):
        yield unicode(self.value)


def format_attributes(**attributes):
    return ' '.join('{key}="{value}"'.format(key=key, value=value)
                    for key, value in attributes.iteritems())


def format_style(**style):
    def format_key(key):
        return key.replace('_', '-')

    def format_value(value):
        if isinstance(value, (tuple, list)):
            return ' '.join(str(_) for _ in value)
        else:
            return repr(value)

    return ';'.join(
        '{key}:{value}'.format(
            key=format_key(key),
            value=format_value(value)
        )
        for key, value in style.iteritems()
    )


class Tag(object):
    name = None
    children = ()
    attributes = {}

    def __init__(self, *children, **attributes):
        self.children = [child if isinstance(child, Tag) else Const(child)
                         for child in children]
        self.attributes = attributes

    def dump(self):
        if self.attributes:
            yield '<{name} {attributes}>'.format(
                name=self.name,
                attributes=format_attributes(**self.attributes)
            )
        else:
            yield '<{name}>'.format(name=self.name)
        yield [_.dump() for _ in self.children]
        yield '</{name}>'.format(name=self.name)

    def dumps(self, indent=0):
        def flatten(dump, indent):
            for item in dump:
                if isinstance(item, basestring):
                    yield item
                else:
                    for subdump in item:
                        for line in flatten(subdump, indent):
                            yield ' ' * indent + line

        lines = flatten(self.dump(), indent)
        if indent > 0:
            return '\n'.join(lines)
        return ''.join(lines)

    def _repr_pretty_(self, printer, cycle):
        printer.text(self.dumps(indent=2))

    def _repr_html_(self):
        return self.dumps(indent=0)


class table(Tag):
    name = 'table'


class tr(Tag):
    name = 'tr'


class td(Tag):
    name = 'td'


class span(Tag):
    name = 'span'


def join_continuous_words(words):
    join = []
    previous = None
    stride = []
    for word, correct in words:
        if previous is None or correct == previous:
            stride.append(word)
        else:
            join.append((' '.join(stride), previous))
            stride = [word]
        previous = correct
    join.append((' '.join(stride), previous))
    return join


def diff_transcripts(guess, etalon):
    stemmer = SnowballStemmer('russian')
    normalize = lambda word: stemmer.stem(word.lower())
    words = set()
    for option in guess:
        for word in option.text.split():
            words.add(normalize(word))
    text = etalon[0].text
    misses = []
    for word in text.split():
        misses.append((word, (normalize(word) in words)))
    words = {normalize(_) for _ in text.split()}
    excesses = []
    if guess:
        text = guess[0].text
        for word in text.split():
            excesses.append((word, (normalize(word) in words)))
    return join_continuous_words(excesses), join_continuous_words(misses)


def format_diff(diff):
    format = []
    for text, correct in diff:
        if correct:
            format.append(text)
        else:
            format.append(' ')
            format.append(
                span(
                    text,
                    style=format_style(border_bottom=('1px', 'solid', RED))
                )
            )
            format.append(' ')
    return format


def show_transcripts(guess, etalon):
    rows = []
    for (guess_segment, guess_part), (etalon_segment, etalon_part) in zip(guess, etalon):
        assert guess_segment == etalon_segment
        guess_diff, etalon_diff = diff_transcripts(guess_part, etalon_part)
        row = []
        if guess_part:
            row.append(
                td(
                    u'— ',
                    *format_diff(guess_diff),
                    style=format_style(border=0)
                )
            )
        else:
            row.append(td(style=format_style(border=0)))
        row.append(
            td(
                u'— ',
                *format_diff(etalon_diff),
                style=format_style(border=0)
            )
        )
        rows.append(tr(*row, style=format_style(border=0)))
    html = table(*rows, style='border:0')
    return html
