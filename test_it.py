import mono_sine_wave
import muco

def test_single_sine_wave_note():
    original_notes = [('A4', 1)]
    wav = mono_sine_wave.make_wav(notes)
    assert muco.recognize(wav) == original_notes

