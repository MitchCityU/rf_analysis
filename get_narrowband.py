#!/usr/bin/env python3
from gnuradio import gr
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio.fft import window 

class SignalExtractor(gr.top_block):
    def __init__(self, input_file, output_file,
                 input_center_freq, target_freq,
                 input_samp_rate, output_samp_rate,
                 bandwidth, duration_sec):
        gr.top_block.__init__(self)

        total_samples = output_samp_rate * duration_sec
        offset_freq = target_freq - input_center_freq

        taps = firdes.complex_band_pass(
            1.0,                     # gain
            input_samp_rate,         # sampling frequency
            -bandwidth / 2,          # lower cutoff
            bandwidth / 2,           # upper cutoff
            5000,                    # transition width
            window.WIN_HAMMING       # correct window enum
        )

        # Blocks
        self.source = blocks.file_source(gr.sizeof_gr_complex, input_file, False)

        self.freq_shift = filter.freq_xlating_fir_filter_ccc(
            1,              # decimation
            taps,
            offset_freq,
            input_samp_rate
        )

        self.resample = filter.rational_resampler_ccc(
            128,            # interpolation
            300,            # decimation
            [],             # taps (empty list = default)
            0.4             # fractional bandwidth
        )

        self.head = blocks.head(gr.sizeof_gr_complex, total_samples)
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, output_file)

        # Connect flowgraph
        self.connect(self.source, self.freq_shift, self.resample, self.head, self.sink)

if __name__ == '__main__':
    input_file = '/mnt/c/Users/15415/Downloads/middle_meteor.iq'
    output_file = '/mnt/c/Users/15415/Downloads/processed_meteor.iq'

    input_center_freq = 137_900_000       # Hz  137.9,   138_140_000
    target_freq = 137_900_000             # Hz
    input_samp_rate = 2_400_000           # Hz (input sample rate)
    output_samp_rate = 1_024_000          # Hz (output sample rate)
    bandwidth = 900                   # Hz
    duration_sec = 30                     # seconds

    tb = SignalExtractor(input_file, output_file,
                         input_center_freq, target_freq,
                         input_samp_rate, output_samp_rate,
                         bandwidth, duration_sec)
    tb.run()   