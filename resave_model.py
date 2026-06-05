import argparse
import os
import sys
import traceback

try:
    import keras
    from keras.layers import Dense as KerasDense
    from keras.utils import register_keras_serializable
except ImportError:
    keras = None
    KerasDense = None
    register_keras_serializable = None

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow tidak terpasang. Silakan install tensorflow terlebih dahulu.")
    sys.exit(1)


def build_custom_objects():
    custom_objects = {}
    if KerasDense is not None and register_keras_serializable is not None:
        @register_keras_serializable(package='keras.layers', name='Dense')
        class DenseWithQuantization(KerasDense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.quantization_config = quantization_config

            def get_config(self):
                config = super().get_config()
                config.pop('quantization_config', None)
                return config

        custom_objects['Dense'] = DenseWithQuantization
    return custom_objects


def resolve_loader():
    if keras is not None:
        return keras.models
    return tf.keras.models


def main():
    parser = argparse.ArgumentParser(description="Muat ulang dan simpan ulang model Keras .keras")
    parser.add_argument(
        "--input",
        default=os.path.join("dashboard", "mindbalance_model_new.keras"),
        help="Path ke file model input .keras",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("dashboard", "mindbalance_model_new_resaved.keras"),
        help="Path output file model .keras yang disimpan ulang",
    )
    parser.add_argument(
        "--no-optimizer",
        action="store_true",
        help="Simpan model tanpa optimizer untuk ukuran yang lebih kecil dan kompatibilitas inferensi.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print("=== Resave Keras Model ===")
    print("Python executable:", sys.executable)
    print("TensorFlow version:", tf.__version__)
    if keras is not None:
        print("Keras version:", keras.__version__)
    else:
        print("Keras standalone tidak terpasang; akan menggunakan tf.keras.")
    print("Input model:", input_path)
    print("Output model:", output_path)
    print()

    if not os.path.exists(input_path):
        print(f"ERROR: File model input tidak ditemukan: {input_path}")
        sys.exit(2)

    custom_objects = build_custom_objects()
    loader = resolve_loader()

    try:
        model = loader.load_model(input_path, compile=False, custom_objects=custom_objects)
    except Exception as exc:
        print("ERROR: Gagal memuat model. Detail exception:")
        traceback.print_exc()
        sys.exit(3)

    print("Model berhasil dimuat.")
    print("Model summary:")
    try:
        model.summary()
    except Exception:
        print("(summary gagal ditampilkan)" )

    save_kwargs = {}
    if args.no_optimizer:
        save_kwargs["include_optimizer"] = False

    try:
        model.save(output_path, **save_kwargs)
    except Exception as exc:
        print("ERROR: Gagal menyimpan ulang model. Detail exception:")
        traceback.print_exc()
        sys.exit(4)

    print("Model berhasil disimpan ulang ke:", output_path)
    print("Selesai.")


if __name__ == "__main__":
    main()
