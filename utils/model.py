
from utils.all_utils import model_cls

LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD"
METRICS = ["accuracy"]

model_clf=model_cls()
model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

"""By default batch_size = 32
"""
def train(X_valid,y_valid,X_train, y_train):
    EPOCHS = 30
    VALIDATION = (X_valid, y_valid)

    history = model_clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, batch_size=32)

def evaluate(X_test, y_test):
    model_clf.evaluate(X_test, y_test)

def save_model():
    model_clf.save("model.h5")