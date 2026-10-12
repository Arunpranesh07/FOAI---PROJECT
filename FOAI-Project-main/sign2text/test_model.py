import pickle
import numpy as np

MODEL_PATH = 'model.p'


def main():
    try:
        m = pickle.load(open(MODEL_PATH, 'rb'))
        print('Loaded object type:', type(m))
        if isinstance(m, dict):
            print('Keys in model dict:', m.keys())
            model = m.get('model') or m.get('clf') or None
        else:
            model = m

        if model is None:
            print('No model object found in pickle. Inspect the file manually.')
            return

        # Inspect predict signature
        print('Model type:', type(model))
        try:
            # Create a dummy input - many models expect shape (n_samples, n_features)
            dummy = np.zeros((1, 126))  # 21 landmarks * 2 coords = 42 -> here we use a bigger safe size
            pred = model.predict(dummy)
            print('Predict succeeded, output:', pred)
        except Exception as e:
            print('Predict failed:', e)

        try:
            proba = model.predict_proba(dummy)
            print('Predict_proba succeeded, sample probabilities shape:', np.asarray(proba).shape)
        except Exception as e:
            print('predict_proba failed or not available:', e)

    except Exception as e:
        print('Failed to load model.p:', e)


if __name__ == '__main__':
    main()