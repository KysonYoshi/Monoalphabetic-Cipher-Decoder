import random
from math import log10
#######################################################################################################################
# 計算information entropy
class ngram_score(object):
    def __init__(self,ngramfile,sep=' '):
        ''' load a file containing ngrams and counts, calculate log probabilities '''
        self.ngrams = {}
        for line in open(ngramfile):
            key,count = line.split(sep)
            self.ngrams[key] = int(count)
        self.L = len(key)
        self.N = sum(iter(self.ngrams.values()))
        for key in self.ngrams.keys():
            self.ngrams[key] = log10(float(self.ngrams[key])/self.N)
        self.floor = log10(0.01/self.N)

    def score(self,text):
        ''' compute the score of text '''
        score = 0
        ngrams = self.ngrams.__getitem__
        for i in range(len(text)-self.L+1):
            if text[i:i+self.L] in self.ngrams: score += ngrams(text[i:i+self.L])
            else: score += self.floor
        return score

########################################################################################################################
# 把依照使用頻率排序的字典load進english_words
def load_words():
    with open('words-by-frequency.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words
english_words = load_words()

########################################################################################################################
# ciphertext sample
#ciphertext='IQYSMEOTZIMYOSIMIQIDGPIVPIVIBIGOEOCIAIVUEOTZPYVEWZVZJZQQOYSIQEWYNMWEWZAZIEWCEIVWICJZZSAZCEVYGZAOTFZVOIQEVYYFCWIHZAVOHZSEWZVZJZQPYVKZCPVYTEWZOVWOAAZSJICZISAFNVCNZAEWZTIKVYCCEWZMIQIDGZHIAOSMEWZAVZIAZAOTFZVOIQCEIVPQZZEIMVYNFYPPVZZAYTPOMWEZVCQZAJGQNUZCUGBIQUZVWICZCEIJQOCWZAISZBCZKVZEJICZYSEWZVZTYEZOKZBYVQAYPWYEWEWZZHOQQYVAAIVEWHIAZVYJCZCCZABOEWPOSAOSMGYNSMCUGBIQUZVWICAOCFIEKWZAEWYNCISACYPVZTYEZFVYJZCOSEYEWZPIVVZIKWZCYPCFIKZ'
#ciphertext='HKMELRHYGGYHENLRYAMREHMAMRVNLREJMHKMVANFMRRERYFNGGYBNAYHENLDEHKHKMESSMUEYHMFNSSOLEHEMRELGNFYGRVYFMRHKMELRHYGGYHENLRFNLRERHMUNWHDNGYCMARNWRNOLUYRMAEMRNWFNSVNREHENLRFAMYHMUWANSGOGGYBEMRFNLHAEBOHMUBCWYHKMARWANSHKMESSMUEYHMFNSSOLEHCYLUYRMHNWSNHENLYFHEJYHMURNOLURHYHENLRYOUEMLFMRVAMRMLFMVANSVHRSOGHEHOUMRNWRHNAEMRHNOLWNGUYOUENUNFOSMLHYAEMRFAMYHMUELFNLJMARYHENLDEHKWYHKMARNLWMUMAYGVANBYHENLELRHYGGYHENLRYAMYLNVMLFYGGHNHKMFNSSOLEHCHKANOZKYVYAHEFEVYHNACGNFYHENLBYRMUFNLHAEBOHNACYOUENYOZSMLHMUAMYGEHCVGYHWNASHKYHFYLBMMIVMAEMLFMUNLGELMHKANOZKYSYVYLUELVKCREFYGRVYFMNABCDYGPELZHKMLMEZKBNAKNNURYRYLYOZSMLHMUYOUENYVVHKMSYVFAMYHMRYRNLEFAMVAMRMLHYHENLNWSMSNAEMRGOGGYBEMRJERENLRYLUUMREAMRFNFAMYHELZLMDFNGGMFHEJMRNFEYGSMSNAEMR'
#ciphertext='ROBEKSCLROSBCPBPXKFKVZYKSPOLXBECLAKSPQZSOPKSCKPZQDZLFKSPOBCZLPOLXYSKPKLBOBCZLPBEKOQBKSLZZLVZLRKFKLBWCVVOVPZDZLPCXKSBEKOYYOSKLBDZLBSOXCDBCZLPCLEKSKLBCLBEKPKQZSGPWECDEOSKIZBEQCLOLDCOVCMKXOLXDZZYKSOBCFKEZWBEKPKSKVOBKBZBEKDZLBSOXCDBCZLPCLFZVFKXCLBSOXCBCZLOVCLPBCBNBCZLOVGZXKVPOLXEZWBEKPKDZLBSOXCDBCZLPGCREBYVOTZNBCLBEKQNBNSK'
#ciphertext='NOGLKWPTNOWGPEGEIKUKBQRKWEOTIGLPTYKWEVQWOEKWPKEQVDQTUKWEOGPQTEOTIRWKEKTGOGPQTEGLKOVGKWTQQTBQTNKUKTGZPBBOBEQDQTEPIKWGLKORROWKTGDQTGWOIPDGPQTEPTLKWKTGPTGLKEKVQWAEZLPDLOWKHQGLVPTOTDPOBPCKIOTIDQQRKWOGPUKLQZGLKEKWKBOGKGQGLKDQTGWOIPDGPQTEPTUQBUKIPTGWOIPGPQTOBPTEGPGSGPQTOBAQIKBEOTILQZGLKEKDQTGWOIPDGPQTEAPNLGRBOXQSGPTGLKVSGSWK'
ciphertext='YMVUNKYTLLTYUWNKTJVKUYVJVKSWNKUHVYMVSJWZVKKUKTZWLLTEWJTYUWNXUYMYMVUDDVGUTYVZWDDONUYUVKUNLWZTLKSTZVKYMVUNKYTLLTYUWNKZWNKUKYVGWFYXWLTPVJKWFKWONGTKVJUVKWFZWDSWKUYUWNKZJVTYVGFJWDLOLLTEUVKZWNYJUEOYVGEPFTYMVJKFJWDYMVUDDVGUTYVZWDDONUYPTNGTKVYWFDWYUWNTZYUHTYVGKWONGKYTYUWNKTOGUVNZVKSJVKVNZVSJWDSYKDOLYUYOGVKWFKYWJUVKYWONFWLGTOGUWGWZODVNYTJUVKZJVTYVGUNZWNHVJKTYUWNXUYMFTYMVJKWNFVGVJTLSJWETYUWNUNKYTLLTYUWNKTJVTNWSVNZTLLYWYMVZWDDONUYPYMJWOAMTSTJYUZUSTYWJPLWZTYUWNETKVGZWNYJUEOYWJPTOGUWTOADVNYVGJVTLUYPSLTYFWJDYMTYZTNEVVBSVJUVNZVGWNLUNVYMJWOAMTDTSTNGUNSMPKUZTLKSTZVWJEPXTLCUNAYMVNVUAMEWJMWWGKTKTNTOADVNYVGTOGUWTSSYMVDTSZJVTYVKTKWNUZJVSJVKVNYTYUWNWFDVDWJUVKLOLLTEUVKHUKUWNKTNGGVKUJVKZWZJVTYUNANVXZWLLVZYUHVKWZUTLDVDWJUVK'
#ciphertext='HYUTUTUKCEZYJWULUTKJPUBJVLEJQUZMUHYCHCTUZJHUCBEQAMCNHRTUPULUZWEHYEZHYUVTCKUTUMUZHPUMCPUBYCLUBUUZCZEZMTUCBUEZCETXJTZULEJQUZMUHANEVEUPXAHYUUGHUZBELURBUJVMYQJTEZUICBCZPJHYUTCETXJTZUMYUKEMCQBCICEZBHMELEQECZNJNRQCHEJZBEZHYUMJZHUGHJVHYUBATECZMELEQWCTEZMTUCBEZIQAHUCTICBEBRBUPHJPEBNUTBUMELEQECZBJVHUZICHYUTUPEZNUCMUVRQNTJHUBHWYEQUCUTECQYUTXEMEPUBPUBHTJACTCXQUQCZPCZPPEBNQCMUCITEMRQHRTCQMJKKRZEHEUBCZPQCTIUBMCQUCTBJZUTCPEMCHUBVJTUBHBHJMTUCHUEZPRBHTECQNQCZHCHEJZBIUZUTCHEZILCBHCZPPCKCIEZIBKJFUMQJRPBKJXEQEOUPXABHCHUCZPMJTNJTCHUNJWUTBHJGEMMQJRPBCVVUMHHYUCETWUXTUCHYUCMTJBBPEVVUTUZHBMCQUBCZPPRTCHEJZBVTJKRTXCZBDRCTUBHJMJZHEZUZHBKJKUZHCTAEZMEPUZHBHJUNJMYCQQCHUZMEUBHYUBUMQJRPBCTUZJHJZQAKUHUJTJQJIEMCQXRHNJQEHEMCQULUZHBBRXSUMHHJPUXCHUCZPMJZHUBHCHEJZRZQEFUFEZUHEMLEJQUZMUWYUTUCBEZIQUQEZUMCZXUPTCWZXUHWUUZCLEMHEKCZPCBKJFEZIIRZEZCZCQAOEZICETXJTZULEJQUZMUMCRBCQEHAEBYCTPHJPUKJZBHTCHUEZHYUBHRPAJVMQJRPBHYUMJZHCMHCZPHYUHTCMUPTEVHCNCTHMCTTEUPCWCAXAWEZPBJTJMUCZMRTTUZHBPEVVRBUPEZHJHYUCHKJBNYUTUMQJRPBCTUHTCZBVJTKCHEJZUKXJPEUPHYUETPAZCKEMBUQRBELUIJLUTZUPXAZJZQEZUCTXUYCLEJTCZPKRQHEMCRBCQQJIEMB'
#ciphertext='NCZRYDNILLINRMYDIJZDRNZJZDEMYDRTZNCZEJMQZDDRDIQMLLIAMJINRMYWRNCNCZRXXZBRINZQMXXFYRNRZDRYLMQILDEIQZDNCZRYDNILLINRMYDQMYDRDNZBMSNWMLIPZJDMSDMFYBIDZJRZDMSQMXEMDRNRMYDQJZINZBSJMXLFLLIARZDQMYNJRAFNZBAPSINCZJDSJMXNCZRXXZBRINZQMXXFYRNPIYBIDZNMSXMNRMYIQNRTINZBDMFYBDNINRMYDIFBRZYQZDEJZDZYQZEJMXENDXFLNRNFBZDMSDNMJRZDNMFYSMLBIFBRMBMQFXZYNIJRZDQJZINZBRYQMYTZJDINRMYWRNCSINCZJDMYSZBZJILEJMAINRMYRYDNILLINRMYDIJZIYMEZYQILLNMNCZQMXXFYRNPNCJMFGCIEIJNRQREINMJPLMQINRMYAIDZBQMYNJRAFNMJPIFBRMIFGXZYNZBJZILRNPELINSMJXNCINQIYAZZHEZJRZYQZBMYLRYZNCJMFGCIXIEIYBRYECPDRQILDEIQZMJAPWILKRYGNCZYZRGCAMJCMMBDIDIYIFGXZYNZBIFBRMIEENCZXIEQJZINZDIDMYRQJZEJZDZYNINRMYMSXZXMJRZDLFLLIARZDTRDRMYDIYBBZDRJZDQMQJZINRYGYZWQMLLZQNRTZDMQRILXZXMJRZD'

########################################################################################################################
# 導入quadgrams來分析適應度，如果字元數足夠長>110則迭代5次即可。如太短則迭代200次避免只找到局部最佳解
if len(ciphertext)>110:
    iteration = 5
    fitness = ngram_score('quadgrams.txt')
    countlimit=1500
else:
    iteration=200
    fitness = ngram_score('quadgrams.txt')
    countlimit=1500

########################################################################################################################
# 頻率分析
print('Frequency analyze start! ')
stored={}
for asciicode in range(65, 91):
    stored[chr(asciicode)] = 0
for char in ciphertext:
    if char not in stored:
        stored[char]=1
    else:
        stored[char]+=1
stored=sorted(stored.items(), key=lambda x:x[1], reverse=True)
for i in range(len(stored)):
    stored[i]=tuple([stored[i][0], stored[i][1]*100/len(ciphertext)])
statistic=list('etaoinshrdlcumwfgypbvkjxqz')# 英文字母出現頻率(statistical data from wiki)
frequency=[]
for i in range(26):
    frequency.append(stored[i][0])

key= {'a':'a'}
max=-99e9 #可讀性初始化
plaintext=""

for i in range(len(frequency)):
    key[frequency[i]] = statistic[i]
p = list(ciphertext)
for i in range(len(ciphertext)):
    p[i] = key[ciphertext[i]]
p = "".join(p).upper()
if fitness.score(p)>max:
    max=fitness.score(p)
    plaintext=p
print(statistic)
print(frequency)
print("Frequency analyze result: ", plaintext.lower(), max) # 輸出經過頻率分析後的"可能"明文

########################################################################################################################
# 開始嘗試對調字母，利用類似information entropy的方式計算文章可讀性，如對調後可讀性越高則繼續對調下去

#   1.將頻率分析後的結果給parentkey，並計算可讀性d1。
#   2.隨機交換parentkey中的兩個字母得到子密鑰child，解密出對應的明文，並計算可讀性d2。
#   3.若d1<d2，則child成為新的parentkey。
#   4.循環步驟二、三 1500次，結束後利用字典檢查是否找得到前五個英文字，若無法則代表非正解，重新迭代一次。
#   5.回到步驟一重新迭代，避免只找到局部最佳解。
#   quadgram statistics 利用information entropy計算可讀性。
parentscore = -99e9
maxscore = -99e9
parentkey=frequency
final=''
while iteration:
    cou=0
    for i in statistic:
        key[parentkey[cou]] = i
        cou+=1
    decipher = ciphertext
    for i in range(len(decipher)):
        decipher = decipher[:i] + key[decipher[i]] + decipher[i + 1:]
    parentscore = fitness.score(decipher)
    count = 0
    while count < countlimit:  # 進行1500次的可讀性计算，預計可找到最佳解
        '''  可讀性計算與比較  '''
        a = random.randint(0,25)        # 隨機交換parentkey中的兩個字母
        b = random.randint(0,25)
        child = parentkey[:]
        child[a], child[b] = child[b], child[a]
        childkey = {'A':'A'}
        for i in range(len(child)):
            childkey[child[i]] = chr(ord('A') + i)
        decipher = ciphertext
        for i in range(len(decipher)):
            decipher = decipher[:i] + childkey[decipher[i]] + decipher[i+1:]
        score = fitness.score(decipher)
        if score > parentscore:         # d1<d2，則child成為新的parentkey。
            parentscore = score
            parentkey = child[:]
            count = 0
        if count==countlimit-1:  #1500次後判萬結果可不可讀
            for i in range(len(parentkey)):
                key[parentkey[i]] = chr(ord('A') + i)
            deci = ciphertext
            for i in range(len(deci)):
                deci = deci[:i] + key[deci[i]] + deci[i + 1:]
            cou=0
            for i in range(20):
                if deci[:i].lower() in english_words:
                    for j in range(20):
                        if deci[i:j+i].lower() in english_words:
                            for x in range(20):
                                if deci[j+i:j + i+x].lower() in english_words:
                                    for y in range(20):
                                        if deci[j + i+x:j + i + x+y].lower() in english_words:
                                            for z in range(20):
                                                if deci[j + i + x+y:j + i+ z+ x + y].lower() in english_words:
                                                    cou += 1
                                                    print(deci[:i], deci[i:j + i], deci[i + j:j + i + x], deci[i + j + x:j + i + x + y], deci[i + j + x+y:j + i + x + y+z], iteration) # 判斷句首是否可讀到五個正確的英文字
                                                    break
            if cou==0:# 若不可讀則重新迭代
                iteration+=1
                parentscore=-99e9
                parentkey = frequency
        count = count + 1

    if (parentscore > maxscore) and cou:       # 若最佳解比歷史最佳解好則記錄下來
        maxscore = parentscore
        maxkey = parentkey[:]
        iteration+=1
        for i in range(len(maxkey)):
            key[maxkey[i]] = chr(ord('A') + i)
        decipher = ciphertext
        for i in range(len(decipher)):
            decipher = decipher[:i] + key[decipher[i]] + decipher[i+1:]
        print("key: " + ''.join(maxkey))
        print("\n")
        final=decipher.lower()
        print("try: " + decipher.lower(), maxscore, iteration, cou)
        print("\n")
    else:
        iteration -= 1
########################################################################################################################
# 輸出解答
print("final plaintext!: ", final)
