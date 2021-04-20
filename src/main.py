from Format import SendFormatResults
from Knowledge import SendKnowledgeResults
from Pruning import SendPruningResults
from TFlite import SendTFliteResults

print('Waiting for the Format experimentation...')
SendFormatResults()
print('Waiting for the TFlite experimentation (including weight clustering and quantization)...')
SendTFliteResults()
print('Waiting for the Pruning experimentation...')
SendPruningResults()
print('Waiting for the Knowledge Distillation experimentation...')
SendKnowledgeResults()
