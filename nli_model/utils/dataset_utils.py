import evaluate

def get_topn_sents(source_sents, summary_sent, topn, score_fn):
    rouge_sents = []

    res = score_fn.compute(predictions=source_sents, references=[summary_sent]*len(source_sents), lang="en", device="cuda:0", batch_size = 32)
    for idx, sent in enumerate(source_sents):
        rouge_sents.append((res["recall"][idx], idx, sent))
    
    # for idx, sent in enumerate(source_sents):
    #     res = score_fn.compute(predictions=[sent], references=[summary_sent])["rougeLsum"]
    #     rouge_sents.append((res, idx, sent))
        
    topn_source_sents = sorted(rouge_sents, key=lambda x: x[0], reverse=True)[:topn]
    topn_source_sents = sorted(topn_source_sents, key=lambda x: x[1])

    topn_scores = [x[0] for x in topn_source_sents]
    topn_source_sent_ids = [x[1] for x in topn_source_sents]
    topn_source_sents = [x[2] for x in topn_source_sents]

    return topn_source_sents, topn_scores, topn_source_sent_ids