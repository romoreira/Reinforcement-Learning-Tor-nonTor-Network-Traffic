from csv import writer

def create_csv(cnn_name):

    List_Exp = ['predict_start', 'predict_end','time_spent_on_prediction', 'predicted_class', 'image_name']
    with open(str(cnn_name)+'_exp_time_spent_on_prediction.csv', 'w') as f:
        writer_object = writer(f)
        writer_object.writerow(List_Exp)
        f.close()

def write_csv(cnn_name, register):
    with open(str(cnn_name) + '_exp_time_spent_on_prediction.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow(register)
        f.close()
        # print("Prediction time recorded")