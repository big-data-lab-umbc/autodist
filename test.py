import matplotlib.pyplot as plt

test_accuracy =  [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.90475, 0.90875, 0.91125, 0.92275, 0.929, 0.92825, 0.92525, 0.93225, 0.9365, 0.93675, 0.94275, 0.94575, 0.93675, 0.93825, 0.94525, 0.95025, 0.94775, 0.9505, 0.949, 0.94875, 0.94925, 0.94875, 0.948, 0.9505, 0.95275, 0.952, 0.9545, 0.953, 0.9545, 0.95225, 0.95325, 0.9545, 0.95225, 0.9525, 0.95375, 0.95425, 0.955, 0.95425, 0.9545, 0.955, 0.954, 0.956, 0.95525, 0.95625, 0.95975, 0.95775, 0.96025, 0.958, 0.95725, 0.96, 0.9585, 0.9585, 0.95875, 0.9615, 0.959, 0.9575, 0.9595, 0.96025, 0.957, 0.9585, 0.96125, 0.95875, 0.9595, 0.96025, 0.962, 0.96075, 0.96025, 0.96125, 0.961, 0.96225, 0.96125, 0.96125, 0.9615, 0.96175, 0.96075, 0.9625, 0.96275, 0.96325, 0.96125, 0.96075, 0.96325, 0.9625, 0.96275, 0.96275, 0.96275, 0.9615, 0.962, 0.9625, 0.96325, 0.96375, 0.96275, 0.963, 0.963, 0.9625, 0.96125, 0.96225, 0.963, 0.963, 0.964, 0.9645, 0.96475, 0.96475, 0.961, 0.9605, 0.963, 0.9655, 0.96475, 0.96525, 0.96525, 0.964, 0.9625, 0.96375]

plt.plot(range(len(test_accuracy)), test_accuracy, 'bo', label='test_accuracy')
#plt.title('Training loss: ' + listToStr)
plt.title('test_accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig("./1average.png")
