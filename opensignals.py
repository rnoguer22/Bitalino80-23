from opensignalsreader import OpenSignalsReader

acq = OpenSignalsReader("Data/ECG_Esther.txt")

acq.raw(2)
acq.signal(2)

acq.raw('ECG')
acq.signal('ECG')

acq.plot('ECG')