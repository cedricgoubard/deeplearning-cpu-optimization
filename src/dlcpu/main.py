from format import SendFormatResults
from knowledge import SendKnowledgeResults
from pruning import SendPruningResults
from tflite import SendTFliteResults

print('Waiting for the Format experimentation...')
SendFormatResults()
print('Waiting for the TFlite experimentation (including weight clustering and quantization)...')
SendTFliteResults()
print('Waiting for the Pruning experimentation...')
SendPruningResults()
print('Waiting for the Knowledge Distillation experimentation...')
SendKnowledgeResults()
