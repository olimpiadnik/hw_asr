import editdistance

def calc_cer(target_text, predicted_text):
    if len(target_text) == 0:
        return 1.0
    
    return editdistance.eval(target_text, predicted_text)/len(target_text)

def calc_wer(target_text, predicted_text):
    if len(target_text) == 0:
        return 1.0
    
    return editdistance.eval(target_text.split(), predicted_text.split())/len(target_text.split())
