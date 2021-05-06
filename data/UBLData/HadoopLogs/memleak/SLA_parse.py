#!/usr/bin/env python
import sys
import xmlrpclib
import socket
import time
import getopt
import os
import thread
#just for sending

def writeLogFile(strmes,logfile,timestamp = False):
    f = open(logfile,'a')
    if (timestamp):
        f.write(str(int(time.time()))+":"+strmes+"\n")
    else:
        f.write(strmes+"\n")
    f.close()
def main():
    progress_score = 0.01
    duration_threshold = 30
    count_threshold = 2
    window_length = 3
    windows = []
    skip_time = 300
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["help"])
    inputfile = "progress.log"
    outputfile = "SLO.log" 
    for opt,arg in opts:
        if opt in ("-h, --help"):
            sys.exit(2)
        if opt in ("-i"):
            inputfile = arg
        if opt in ("-o"):
            outputfile = arg
    if os.path.exists(outputfile):
        os.remove(outputfile)              
    print "Reading Workload File"
    inputF = open(inputfile,"r")
    for i in range(400):
        line=inputF.readline() #ignore the training data
    count = 0
    for i in range(window_length):
        windows.append(0)
    c  = 0
    first_time = 0
    while inputF:
        line=inputF.readline()
        if len(line)<3:
            break
        array = line.split('|')

        if float(array[2]) < progress_score and float(array[1]) < 99.9 and float(array[1]) > 0.1:
            if first_time == 0:
                first_time = int(array[0])
                #print first_time
            else:
                if int(array[0]) - first_time > duration_threshold:
                    
                    print str(array[0])
                    writeLogFile(str(array[0]),outputfile)
                    c += 1
                    first_time = 0
                    for i in range(skip_time):
                        if (inputF):
                            line=inputF.readline()
        else:
            first_time = 0

        

    print "Total:"+str(c)
if __name__ == "__main__":
    main()
