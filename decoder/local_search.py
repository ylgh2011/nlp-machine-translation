import copy

def local_search(sentence, lm):


    def get_prob(test_list):
        stance = []
        for i in test_list:
            stance += (i.split())
        stance = tuple(stance)
        lm_state = (stance[0],)
        score = 0.0
        for word in stance[1:]:
            (lm_state, word_score) = lm.score(lm_state, word)
            score += word_score
        return score


    eng_list = copy.deepcopy(sentence)
    eng_list[:0] = ["<s>"]
    eng_list.append("<s>")

    while True:
        best_list = copy.deepcopy(eng_list)

            # insert
            for i in range(1, len(eng_list) - 1):
                for j in range(1, i):
                    now_list = copy.deepcopy(eng_list)
                    now_list.pop(i)
                    now_list.insert(j, eng_list[i])
                    if get_prob(now_list) > get_prob(best_list):
                        best_list = now_list
                for j in range(i+2, len(eng_list) - 1):
                    now_list = copy.deepcopy(eng_list)
                    now_list.insert(j, eng_list[i])
                    now_list.pop(i)
                    if get_prob(now_list) > get_prob(best_list)
                        best_list = now_list

            # swap
            for i in range(1, len(eng_list) - 2):
                for j in range(i + 1, len(eng_list) - 1):
                    now_list = copy.deepcopy(eng_list)
                    now_list[i], now_list[j] = now_list[j], now_list[i]
                    if get_prob(now_list) > get_prob(best_list):
                        best_list = now_list

            if get_prob(best_list) == get_prob(eng_list):
                break
            else:
                eng_list = best_list

        return ' '.join(eng_list[1:-1])
