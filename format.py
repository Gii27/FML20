def scale_format():
    try:
        with open("abalone.data", 'r') as infile, open("train.data", "w") as outtrain, open("test.data", "w") as outtest:
            lines = infile.readlines()
            count = 0
            for line in lines:
                count = count + 1
                line = line.strip().split(",")
                outline = ""
                label = line[-1]
                outline = outline + label

                outline = outline + " "
                if line[0] == "M":
                    outline += "1:1"
                elif line[0] == "F":
                    outline += "1:-1"
                else:
                    outline += "1:0"

                for i in range(1, len(line) - 1):
                    outline = outline + " "
                    field = str(i + 1) + ":" + line[i]
                    outline = outline + field
                if count <= 3133:
                    print(outline, file=outtrain)
                else:
                    print(outline, file=outtest)
    except IOError as e:
        print("Operation failed: %s" % e.strerror)


def binary_format():
    try:
        with open("abalone.data", 'r') as infile, open("binarytrain.data", "w") as outtrain, open("binarytest.data", "w") as outtest:
            lines = infile.readlines()
            count = 0
            for line in lines:
                count = count + 1
                line = line.strip().split(",")
                outline = ""
                label = line[-1]
                if 1 <= int(label) <= 9:
                    label = "+1"
                else:
                    label = "-1"
                outline = outline + label

                outline = outline + " "
                if line[0] == "M":
                    outline += "1:1"
                elif line[0] == "F":
                    outline += "1:-1"
                else:
                    outline += "1:0"

                for i in range(1, len(line) - 1):
                    outline = outline + " "
                    field = str(i + 1) + ":" + line[i]
                    outline = outline + field

                if count <= 3133:
                    print(outline, file=outtrain)
                else:
                    print(outline, file=outtest)
    except IOError as e:
        print("Operation failed: %s" % e.strerror)


binary_format()
