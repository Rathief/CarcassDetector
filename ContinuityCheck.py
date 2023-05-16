import os

def CheckContinuity(track_dir):
    output = []
    for filename in os.listdir(track_dir):
        filepath = os.path.join(track_dir, filename)
        if os.path.isfile(filepath):
            trackText = open(filepath, 'r')
            lineArray = []
            for line in trackText:
                lineArray.append(line)
            trackText.close()

            current_frame = int(lineArray[0].split()[0])
            emptyFrame = []
            for line in lineArray:
                next_frame = int(line.split()[0])
                if next_frame - current_frame > 1:
                    emptyFrame.append(list(range(current_frame+1, next_frame)))
                current_frame = next_frame
            if len(emptyFrame) > 0:
                output.append([filename, len([element for innerList in emptyFrame for element in innerList]), emptyFrame])
    print(*output, sep='\n')


CheckContinuity("track")