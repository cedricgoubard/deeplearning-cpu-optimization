#from format import SendFormatResults
from knowledge import send_knowledge_results
from pruning import send_pruning_results
from tflite import send_tflite_results

#SendFormatResults()
send_tflite_results()
send_pruning_results()
send_knowledge_results()
