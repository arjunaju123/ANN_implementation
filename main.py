from utils.all_utils import prep_data
from utils.model import train,evaluate,save_model
import matplotlib.pyplot as plt
import logging
import os

logging_str = "[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
log_dir="logs"
os.makedirs(log_dir, exist_ok=True)
#%(asctime)s: =>Time in which code is executed 
#%(levelname)s: => Whcih information are you getting ..Is it the debug information ..or error information..or exception information etc
#%(module)s: =>Which module has raised the error information
#%(message)s: =>What message to print..eg:if the training is started ...if there is any exceptions 
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode='a')
#filemode='a'used to append all the logs after each run
#basic configuration is set only in the main file. We are calling all the modules in here.
def main(X_new,Y_pred,y_test):

  for img_array, pred, actual in zip(X_new, Y_pred, y_test[:3]):
    plt.imshow(img_array, cmap="binary")
    plt.title(f"predicted: {pred}, Actual: {actual}")
    plt.axis("off")
    plt.show()
    print("---"*20)

if __name__=='__main__':

     result=prep_data()

     try:
        logging.info(">>>>> Starting training, evaluation and saving the model >>>>>")
        train(result['X_valid'],result['y_valid'],result['X_train'],result['y_train'])
        evaluate(result['X_test'],result['y_test'])
        save_model()
        main(result['X_new'],result['Y_pred'],result['y_test'])
        logging.info("<<<<< All process done successfully <<<<<\n")
     except Exception as e:
        logging.exception(e)
        raise e