class Progressbar:
    def __init__(self, goal, steps=5, lenght=50, name=''):
        self.goal=goal
        self.steps=steps
        self.done=0
        self.name=name
        self.lenght = lenght

        self.step_size = goal/steps


    def update(self, i):
        if i%10==0:
            while i>=self.done*self.step_size:
                self.message()
                self.done+=1

    def message(self):
        bar = '[' + '='*round(self.done*self.lenght/self.steps) + '>' + '-'*(self.lenght-round(self.done*self.lenght/self.steps)) + ']'

        '''    
        bar ='['
        for _ in range(round(self.done*self.lenght/self.steps)):
            bar+='='
        bar+=">"
        for _ in range(self.lenght-round(self.done*self.lenght/self.steps)):
            bar+= '-'
        bar+=']'
        '''

        print(bar, self.name, round(100*self.done/self.steps), "%")

    def __del__(self):
        self.message()
        #print(self.name, "finished!")