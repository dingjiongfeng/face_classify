import pandas as pd

# submission = pd.read_csv('submission/submission.csv', sep='\t')
# print(submission.head())
if __name__ == '__main__':
    dfs = pd.DataFrame([['1.jpg', 1], ['2.jpg', 0]],
                       columns=['fnames', 'label'])

    result = dfs['fnames']+'\t'+dfs['label'].map(str)
    result = pd.DataFrame(result, columns=['fnames\tlabel'])
    print(result)
