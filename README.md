# Генерация музыки с заранее заданной эмоцией
Репозиторий посвящен экспериментам с контролируемой генерацией музыки с заранее заданной эмоцией.

Ноутбук emotion_generation.ipynb содержит инференс, в процессе которого осуществляется генерация эмоциональной музыки.

Файл emotion_classification/linear_clf/linear_classifier.py содержит классификацию MIDI на эмоции. 
Классиикация осуществляется на основе данных из music_generation/data/music_midi/emotion_midi_text_neo

Для того, чтобы преобразовать id токенов, выдаваемых моделью, в midi, нужно открыть ноутбук [https://colab.research.google.com/drive/10ZAdEwHDbL1lVcUGeCdj9FxXnQSNFSH4?usp=sharing](https://colab.research.google.com/drive/10ZAdEwHDbL1lVcUGeCdj9FxXnQSNFSH4?usp=sharing), запустить все ячейки и добавить в конце следующий код:
```
et = mmm.EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER
enc = mmm.getEncoder(et)

enc.tokens_to_midi(tokens, <path/to/midifile>)  # tokens - id токенов из модели
```
