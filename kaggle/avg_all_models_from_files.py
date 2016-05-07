from collections import defaultdict, Counter
from glob import glob

def avg_all_models_from_files(infiles, filename):
    '''
    Takes three output files procuded by the final_predictions.py and used hard voting to produce the final result.
    This is the 3rd layer of our classifier.
    '''
    scores = defaultdict(list)
    with open(filename,"wb") as outfile:
        weight_list = [1]*len(glob(infiles))
    
        for i, ifile in enumerate( glob(infiles) ):
            print "parsing:", ifile
            lines = open(ifile).readlines()
            lines = [lines[0]] + sorted(lines[1:])
        
            #write out all model results
            for idx, line in enumerate( lines ):
                if i == 0 and idx == 0:
                    outfile.write(line)
                    
                if idx > 0:
                    row = line.strip().split(",")
                    for l in range(1,weight_list[i]+1):
                        scores[(idx,row[0])].append(row[1])
    
        #take hard votes
        for j,k in sorted(scores):
            outfile.write("%s,%s\n"%(k,Counter(scores[(j,k)]).most_common(1)[0][0]))
    
        print("wrote to %s" % outfile)

if __name__ == '__main__':
        avg_all_models_from_files("stack*","output.csv")

