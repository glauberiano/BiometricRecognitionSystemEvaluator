def bla(row, ):
    match_sum = 0
    previousDimMatched = False
    score=list()
    for dim in user_model.model.keys():
        if (row[dim] <= user_model.model[dim][1]) and (row[dim] >= user_model.model[dim][0]):
            if previousDimMatched:
                match_sum = match_sum + 1.5
            else:
                match_sum = match_sum + 1.0
            previousDimMatched = True
        else:
            previousDimMatched = False
    #import pdb;pdb.set_trace();
    score = match_sum/max_sum
    return score


test_stream.apply(bla(x),axis=1)


        # def func1(row, user_model):
        #     match_sum = 0
        #     previousDimMatched = False
        #     for dim in user_model.model.keys():
        #         if (row[dim] <= user_model.model[dim][1]) and (row[dim] >= user_model.model[dim][0]):
        #             if previousDimMatched:
        #                 match_sum = match_sum + 1.5
        #             else:
        #                 match_sum = match_sum + 1.0
        #             previousDimMatched = True
        #         else:
        #             previousDimMatched = False
        #     #import pdb;pdb.set_trace();
        #     score = match_sum/max_sum
        #     return score
        # scores = test_stream.apply(lambda x : func1(x, user_model), axis=1)

        # assert len(scores) == len(test_stream), 'scores esta errado'
        # #import pdb;pdb.set_trace();
        # flag_score = [1 if score > decision_threshold[genuine_user] else 0 for score in scores]
        # flag_y_true = [1 if user == genuine_user else 0 for user in y_test]

        # if self.adaptive == False:
