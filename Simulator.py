# read a tracking result file
# find average position and size
# create new file with a simulated carcass using avg pos and size as attribute
# sim carcass obj num is always 0
import os


def simulate_carcass(track_dir, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for filename in os.listdir(track_dir):
        filepath = os.path.join(track_dir, filename)
        if os.path.isfile(filepath):
            trackText = open(filepath, 'r')
            trackTextArray = []
            lineCount = 0
            for line in trackText:
                lineCount += 1
                trackTextArray.append(line.split())
            trackText.close()

            avgPos = [0, 0]
            avgSize = [0, 0]
            for line in trackTextArray:
                avgPos[0] += int(line[2])
                avgPos[1] += int(line[3])
                avgSize[0] += int(line[4])
                avgSize[1] += int(line[5])
            avgPos[0] /= lineCount
            avgPos[1] /= lineCount
            avgSize[0] /= lineCount
            avgSize[1] /= lineCount

            simulatedTrack = open(os.path.join(save_dir, "Simulated" + filename), 'w')
            currentFrame = 0
            simAttribute = list(map(round, avgPos + avgSize))
            for line in trackTextArray:
                # if the video progresses, insert virtual object
                if int(line[0]) > currentFrame:
                    currentFrame = int(line[0])
                    simulatedTrack.write(' '.join(map(str, [currentFrame, 0] + simAttribute + [1, '\n'])))

                simulatedTrack.write(' '.join(line[0:6] + ['0', '\n']))
            simulatedTrack.close()

    # print(avgPos)
    # print(avgSize)
    # print(' '.join(map(str, [currentFrame, 0] + simAttribute + [-1, -1, -1, 0])))


if __name__ == '__main__':
    simulate_carcass("track", "Simulated_Track")
