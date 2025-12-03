# src/modeling.py
import tensorflow as tf


def build_and_train_resnet50(
    train_ds, val_ds,
    num_classes,
    label_mode="categorical",
    unfreeze_blocks=2,
    warmup_epochs=1,
    finetune_epochs=1,
    base_lr=1e-3,
    ft_lr=1e-5,
    use_label_smoothing=True,
    class_weight=None
):
    base = tf.keras.applications.resnet.ResNet50(
        include_top=False, input_shape=(224, 224, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    

    act = "softmax" if (label_mode == "categorical" and num_classes > 1) else (
        "sigmoid" if num_classes == 1 else "softmax"
    )
    outputs = tf.keras.layers.Dense(num_classes, activation=act)(x)
    model = tf.keras.Model(inputs, outputs)

    # --- Loss + metrics
    if label_mode == "categorical":
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1 if use_label_smoothing else 0.0
        )
        acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    # If F1 throws error in your TF version, comment it out or use tf-addons
    metrics = [acc_metric]  # , tf.keras.metrics.F1Score(average="weighted")]

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=3, factor=0.5, min_lr=1e-6, monitor="val_loss"
        ),
    ]

    # --- Phase 1: warmup
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=1e-4),
        loss=loss,
        metrics=metrics,
    )
    warmup_hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=cbs,
        class_weight=class_weight,
    )

    # --- Phase 2: unfreeze top blocks
    for layer in base.layers:
        if any(tag in layer.name for tag in [f"conv5_", f"conv4_"][:unfreeze_blocks]):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=ft_lr, weight_decay=1e-4),
        loss=loss,
        metrics=metrics,
    )
    finetune_hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=cbs,
        class_weight=class_weight,
    )

    return model, {"warmup": warmup_hist, "finetune": finetune_hist}

