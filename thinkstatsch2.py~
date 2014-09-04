import thinkstats
import math
import first
import survey
import Pmf
import operator

#exercise 2.1: compute mean, variance and standard deviation of pumpkins
def Pumpkin(x): 
    themean = thinkstats.Mean(x)
    thevar = thinkstats.Var(x)
    thestdv = math.sqrt(thevar)
    print 'the mean is:', themean
    print 'the variance is:', thevar
    print 'the standard deviation is:', thestdv

#exercise 2.2: compute the std dev of gestation time for first babies and others
def GestationTime():
    table = survey.Pregnancies()
    table.ReadRecords()
    firsts, others = first.PartitionRecords(table)
    firsts.lengths = [p.prglength for p in firsts.records]
    firsts.std = math.sqrt(thinkstats.Var(firsts.lengths))
    others.lengths = [p.prglength for p in others.records]
    others.std = math.sqrt(thinkstats.Var(others.lengths))
    print 'the standard deviation of first babies gestation time is:',firsts.std
    print 'the standard deviation of non-first babies gestation time is:',others.std
    print 'the difference in std devs is:',(firsts.std-others.std)
    firsts.mean = thinkstats.Mean(firsts.lengths)
    others.mean = thinkstats.Mean(others.lengths)
    print 'the mean of first babies gestation time is:',firsts.mean
    print 'the mean of non-first babies gestation time is:',others.mean
    print 'the difference in means is:',(firsts.mean-others.mean)

#exercise 2.3: mode
def Mode(hist):
    values = hist.Values()
    f = 0
    v = 0
    for value in values:
        f1 = hist.Freq(value)
        if f1 > f:
            f = f1
            v = value
    print 'Mode is:', v, ' with a frequency of ', f

def AllModes(hist):
    values = hist.Values()
    freqvals = {}
    for value in values:
        f2 = hist.Freq(value)
        freqvals[value] = f2
    freqvals = sorted(freqvals.iteritems(), key = operator.itemgetter(1), reverse = True)
    print freqvals
        
#exercise 2.5: 
def PmfMean(x):
    mean = 0
    values = x.Values()
    for value in values:
        prob = x.Prob(value)
        mean = mean + prob*value
    print 'pmf mean is:', mean
    return mean

def PmfVar(x):
    var = 0.0
    values = x.Values()
    mean = PmfMean(x)
    for value in values:
        prob = x.Prob(value)
        var = var + (prob*(value-mean)*(value-mean))
    print 'pmf var is:', var

#run these values
if __name__ == '__main__':
    Pumpkin([1,1,1,3,3,591])
    GestationTime()
    hist = Pmf.MakeHistFromList([1,2,2,6,6,6,3,5])
    Mode(hist)
    AllModes(hist)
    pmf = Pmf.MakePmfFromList([1,2,2,3,5])
    PmfMean(pmf)
    PmfVar(pmf)
