from random import randint
import os
class Alien:
    sentence =""
    mood=""

    def __init__(self,s, m):
        self.sentence=s
        self.mood=m
        return
    



def gen(num):
    alph=["meep", "morp"]
    aliens=[]
    for i in range(num):
        b=""
        h=0
        s=0
        l=randint(1,50)
        for x in range(l):
            ind= randint(0,len(alph)-1)
            b+=(alph[ind]+" ")
            if ind==0:
                h+=1
            else:
                s+=1
        
        
        if l>20 and h/l>=0.6: 
            a=Alien(b,"happy")
        else:
            a=Alien(b,"sad")
        #if l%2==0:
         #  a=Alien(b,"happy")
        #else:
          #  a= Alien(b,"sad")

        aliens.append(a)
    return aliens

def save(list, file):
    with open(file, "w", encoding="utf-8") as f:
        for a in list:
            string= a.sentence +"00"+a.mood+"\n"
            f.write(string)
    return

def main():
    aliensList=gen(100)
    save(aliensList,(os.path.join(os.getcwd(),"ForwardPassNN\data\dataBatch.txt")))

main()