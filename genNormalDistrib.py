#!/usr/bin/python
# how many data points in the time sereies
numrows=250000
# go build it
import random
# get now
# timenow=calendar.timegm(time.gmtime())
#
# get repeatable runs
random.seed(10)
# random.gauss(mu, sigma)
#  row content:  userid|temperature|swapping|numberofjobs|sensor3
for x in range(1,numrows):
        myuser=random.randint(1000000, 9000000)
        mytemp=abs(random.gauss(.5,.3 ))
        myswap=abs(random.gauss(.5,.2 ))
        myjobs=abs(random.gauss(.5,.1 ))
        mynop=abs(random.gauss(100,20 ))
        if ( mytemp+myswap+myjobs > 2 and mytemp+myswap+myjobs < 2.5 ):
          myfail=1
        else:
          myfail=0
        print ( "%7i|%10.8f|%10.8f|%10.8f|%10.8f|%1i" %(myuser, mytemp, myswap,myjobs, mynop, myfail) ) 
