import os
import pandas as pd

p = n = 0
if __name__ == '__main__':
    path = "./output/"
    pd_all = pd.read_csv(os.path.join(path, "test_results.tsv") ,sep='\t',header=None)

    data = pd.DataFrame(columns=['label'])
    print(pd_all.shape)

    for index in pd_all.index:
        negative_score = pd_all.loc[index].values[0]
        positive_score = pd_all.loc[index].values[1]

        if max( positive_score, negative_score) == positive_score:
            #data.append(pd.DataFrame([index, "positive"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = [ "positive"]
            p = p+1
        else:
            #data.append(pd.DataFrame([index, "negative"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = [ "negative"]
            n = n+1
        #print(negative_score, positive_score, negative_score)

    data.to_csv(os.path.join(path, "pre_sample.tsv"),sep = '\t')
    #print(data)
    print("正面：%d条,负面：%d条",p,n)
    print("正面百分比：%.2f%%" % ((p / (p + n)) * 100))
