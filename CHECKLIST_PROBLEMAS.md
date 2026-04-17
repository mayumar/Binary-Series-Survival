# Checklist de problemas del proyecto

## Críticos

- [ ] Corregir la generación de ventanas para que no mezclen distintos ciclos `run-to-failure`.
  Referencias: `data_preprocessing/preprocess.py` (`format_and_concat`, generación de ventanas en `preprocess_df`).

- [ ] Alinear correctamente las etiquetas `y` y `c` con cada ventana generada.
  Referencias: `data_preprocessing/compute_dataloaders.py` (`concat_and_discretize`), `data_preprocessing/preprocess.py` (ventanas en `preprocess_df`).

- [ ] Revisar la coherencia entre `X` y `y/c` después del `sort_values("time")` para evitar desincronización de features y labels.
  Referencias: `data_loading/data_loader.py` (`load_data`, `read_csv_by_index_batches_sorted`).

## Altos

- [ ] Arreglar el particionado `train/val/test` cuando hay pocos ciclos para evitar tamaños negativos o splits inválidos.
  Referencia: `data_loading/data_loader.py`.

- [ ] Corregir el `WeightedRandomSampler`, que ahora usa pesos calculados sobre eventos pero aplicados sobre todo el dataset.
  Referencia: `data_preprocessing/compute_dataloaders.py`.

- [ ] Adaptar `models/transformer/trainer_transformer.py` a la firma real de `train()` para que esa ruta funcione.
  Referencias: `models/transformer/trainer_transformer.py`, `models/utils/functions.py`.

- [ ] Añadir manejo seguro cuando validación o test generan cero ventanas, para que la evaluación no rompa.
  Referencias: `data_preprocessing/preprocess.py`, `models/utils/functions.py`.

## Medios

- [ ] Corregir la `ranking_loss` de `DiscreteHazardNLL`, que ahora calcula una CDF inconsistente a partir de hazards.
  Referencia: `models/utils/losses.py`.

- [ ] Hacer que el preprocesado funcione también si no queda ninguna variable numérica tras el filtrado.
  Referencia: `data_preprocessing/preprocess.py`.

- [ ] Aplicar realmente `start_idx` desde la configuración, o eliminarlo si ya no se usa.
  Referencias: `main.py`, `config/config.json`.

- [ ] Revisar el filtro de ciclos cortos `mask.sum() > 2*3600`, porque asume una frecuencia de muestreo fija.
  Referencia: `data_loading/data_loader.py`.

## Operativos

- [ ] Instalar o documentar correctamente las dependencias del entorno, porque ahora faltan al menos `torch` y `pandas` en la ejecución local.
  Referencia: `requirements.txt`.

## Seguimiento

- Estado recomendado por cada punto:
  - `Pendiente`
  - `En progreso`
  - `Bloqueado`
  - `Resuelto`
  - `Verificado`

- Sugerencia de orden de resolución:
  1. Ventanas mezclando ciclos.
  2. Alineación entre ventanas y etiquetas.
  3. Coherencia entre `X` y `y/c`.
  4. Particionado de ciclos.
  5. Sampler y evaluación.
  6. Resto de inconsistencias del pipeline.
