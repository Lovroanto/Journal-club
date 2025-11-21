conda activate journal-club-presentation-generator

so the point is grobid did a a pretty bad job with this nature article displacing a lot of text so i am thinking if ther is a way to use in parallel one other programme we worked and use parts of it to check the work of grobid. Like let s Grobid do it s work and then 
ok so what happened in the main body there are a few divisions only the first one is the article the second and third... were not 
for some reson he took the first two divisions in the main and tehn went to supplementary while i d ont understand why he did not stop at the first one for main is there any way to make that better



Ok great it works. Now i want to add one more feature to the model. When all the analysis is done he does one more thing, the idea behind it: clean has the good order of sentences so we will have to understand how to rewright the order of before. To do that we start with clean 001 and we check that for it s link meaning grobid has the sentence number bigger than the one before him. In the case of 001 there is nothing to check, in the case of clean 002 we need to check that the linked grobid sentence number is bigger that the sentence number of the linked grobid to 001. 

If it is not the case something is on the wrong place, to understand the mistake we will wright in an integer which place it is like clean 58 -> grobid 67 for exemple then integer is 58 and an other integer is 68. THen the programme will look for grobid 69 and check:

 if it has a score non zero to a link that is also not 0. 

If this is the case then we will look at which clean it is (the link and )  