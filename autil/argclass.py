import json
import re


def load_args(file_path):
    ''' Load args/** .json '''
    args_dict = loadmyJson(file_path)
    # print("load arguments:", args_dict)
    args = ARGs(args_dict)
    return args


class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


################################################
class xstr:
    def __init__(self, instr):
        self.instr = instr

    def rmCmt(self):
        qtCnt = cmtPos = 0
        rearLine = self.instr

        while rearLine.find('//') >= 0: # “//”
            slashPos = rearLine.find('//')
            cmtPos += slashPos
            # print 'slashPos: ' + str(slashPos)
            headLine = rearLine[:slashPos]
            while headLine.find('"') >= 0:
                qtPos = headLine.find('"')
                if not self.isEscapeOpr(headLine[:qtPos]):
                    qtCnt += 1
                headLine = headLine[qtPos+1:]
                # print qtCnt
            if qtCnt % 2 == 0:
                # print self.instr[:cmtPos]
                return self.instr[:cmtPos]
            rearLine = rearLine[slashPos+2:]
            # print rearLine
            cmtPos += 2
        # print self.instr
        return self.instr

    def isEscapeOpr(self, instr):
        if len(instr) <= 0:
            return False
        cnt = 0
        while instr[-1] == '\\':
            cnt += 1
            instr = instr[:-1]
        if cnt % 2 == 1:
            return True
        else:
            return False


def loadmyJson(JsonPath):
    try:
        srcJson = open(JsonPath, 'r', encoding= 'utf-8')
    except:
        print('cannot open ' + JsonPath)
        quit()

    dstJsonStr = ''
    for line in srcJson.readlines():
        if not re.match(r'\s*//', line) and not re.match(r'\s*\n', line):
            xline = xstr(line)
            dstJsonStr += xline.rmCmt()

    # print dstJsonStr
    dstJson = {}
    try:
        dstJson = json.loads(dstJsonStr)
    except:
        print(JsonPath + ' is not a valid json file')

    return dstJson
