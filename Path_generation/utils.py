import re
import torch
import pandas as pd
import unicodedata
import re

def freeze_params(model):
    for par in model.parameters():
        par.requires_grad = False


def get_set_relations(data,hop=1):
    '''
    :param data: dataset
    :param hop: hop number
    :return: return unique relations existing in the data
    '''
    tagz =  data['TAG'].apply(lambda x: x.split('|'))
    if hop == 1: return data['TAG'].unique()
    else:
        uniq_rel = set()
        for i in tagz:
            for x in i : uniq_rel.add(x)
        return uniq_rel

def preprocess_data(data):
    data['TAG'] = data['TAG'].apply(lambda x: x.replace('|',' '))
    data['QA'] =  data['QA'].apply(lambda x: re.sub(r'(\[.*\])+', "NE",x))
    return data['QA'].values , data['TAG'].values


def generate_seq(model, input_text,tokenizer, num_beam=3, num_return_sequences=3):
    '''
    Generate sequences for each sample
    :param model: BART model
    :param input_text: input question
    :param tokenizer: tokenizer
    :param num_beam: number of beams
    :param num_return_sequences: number of returned sequences
    :return: a list of generated paths, list of scores : score for each path , list of list of scores for each element in the sequence
    '''
    model.eval()
    ans = []
    scores = []
    hop_scores = []
    # tokenize the question
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to('cuda')
    # generate sequences
    output = model.generate(input_ids, num_beams=3, num_return_sequences=3,
                            early_stopping=True,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                            no_repeat_ngram_size=3,
                            min_length=0)
    transition_scores = model.compute_transition_beam_scores(output.sequences, output.scores, output.beam_indices)
    temp = set()
    dd = torch.softmax(output.sequences_scores, dim=0).tolist()
    for i, beam_output in enumerate(output.sequences):
        ans.append(tokenizer.decode(beam_output, skip_special_tokens=True).strip().rstrip().lower())
        hop_scores.append(transition_scores[i, 1:-1].exp().detach().cpu().tolist())
        if ans[i] not in temp:
            print(i, " : Generated ==> ", ans[i], dd[i])
            temp.add(ans[i])
        scores.append(dd[i])
    return ans, scores, hop_scores


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])",'', w)
    w = re.sub(r'(\[.*\])+', "", w)
    w = re.sub(r"[^a-zA-Z0-9_?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w =w.replace('\\','')
    return w

def run_evaluation(model,data_test,tokenizer,to_exclude,hops):
    results = []
    hits3 = 0
    hits1 = 0
    for i, test in data_test.iterrows():
        print('--', i, "Q: ", test['text'], "Ans: ", test['tag'])
        tmp, scores, hopscores = generate_seq(model, test['text'],tokenizer)
        print(tmp, test['text'])
        results.append((test['text'], '|'.join(tmp), scores, hopscores))
        if test['tag'] in tmp:
            hits3 += 1
        if tmp[0] == test['tag']:
            hits1 += 1
        print(" HITS 3 ", hits3 / (i + 1), "HITS 1 ", hits1 / (i + 1))

    result_data = pd.DataFrame(results, columns=['QA', 'Rel', 'Scores', 'Hopscores'])
    result_data.to_csv('../Data/Path_gen/predictions_metaqa_'+hops+'.txt', sep='\t')

    print('Total hits ', hits3)
    print('Hits@3 ', hits3 / (len(data_test) - to_exclude), 'Hits@1',hits1/ (len(data_test) - to_exclude))