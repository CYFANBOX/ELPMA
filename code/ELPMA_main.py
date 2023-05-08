import ELPMA_utils as ut
import pandas as pd
import pickle


def ELPMA_predict_unknown(data_path):
    unknown, pse, miRNA, pse_name, miRNA_name = ut.import_dataset(data_path)
    unknown_data = []
    for item in unknown:
        temp = pse[item[0], :].tolist() + miRNA[item[1], :].tolist()
        unknown_data.append(temp)

    print('\nLoading model ...\n')
    with open('../code/ELPMA_.pkl', 'rb') as f:
        ELPMA = pickle.load(f)

    print('\nBe predicting ...\n')
    base_preds, base_probs = ut.base_preds_probs(unknown_data, ELPMA)
    pred_final, prob_final = ut.soft_voting(base_probs)

    print('\nSaving results ...\n')
    pse_names = []
    miRNA_names = []
    for item in unknown:
        pse_names.append(pse_name[item[0]][0][0])
        miRNA_names.append(miRNA_name[item[1]][0][0])

    content = {'pseudogene': pse_names, 'miRNA': miRNA_names, 'score': prob_final}
    excel_writer = pd.ExcelWriter('../results/predict_score.xlsx')
    dataScore = pd.DataFrame(content).sort_values(by='score', ascending=False)
    dataScore.to_excel(excel_writer)
    excel_writer.save()


if __name__ == '__main__':
    data_path = '../data/'
    base_learner_num = 10
    ELPMA_predict_unknown(data_path)


