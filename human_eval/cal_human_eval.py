import json
import statsmodels.api as sm
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir for evaluation")
    args = parser.parse_args()

    with open(args.data_file) as infile:
        samples = json.load(infile)

    df = pd.DataFrame.from_dict(samples)
    df.columns = ['interestingness', 'temporality', 'entropy', 'n_after', 'model', 'pred_relations',
                  'generated_text', 'passage_id']

    ft_cols = ['temporality', 'n_after', 'entropy']

    y = df['interestingness']
    X = df[ft_cols]

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    models = ['baseline', 'vanilla', 'structured_prompt', 'pretrained', 'rl']
    if 'wp' in args.data_file:
        models.pop(-2)


    for m in models:
        print('>>>>>', m)
        print("temporality score is %.3f" % df.loc[df['model'] == m, ['temporality']].mean()[0])
        print("interestingness score is %.3f" % df.loc[df['model'] == m, ['interestingness']].mean()[0])


if __name__ == "__main__":
    main()