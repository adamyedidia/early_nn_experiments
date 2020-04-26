def trainingSetParser():
    trainLines = open("train.json", "r").readlines()
    
    
    
    for line in trainLines:
        if "cuisine" in line:
            lineSplit = string.split(line)
            
            cuisine = lineSplit[1]
            
            