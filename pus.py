import numpy as np
from graphics import *
import time


class Wampus:
    GOLD = 2
    ME = 1
    EMPTY = 0
    VAMPIRE = 3
    HOLE = 5
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    MAP_UP = 0
    MAP_DOWN = 1
    MAP_LEFT = 2
    MAP_RIGHT = 3
    ACTIONS = {MAP_UP: UP, MAP_DOWN: DOWN, MAP_LEFT: LEFT, MAP_RIGHT: RIGHT}
    ACTION_NUMBERS = 4

    def __init__(self, width, height, obstacles, me, vampires, gold, q=0):
        self.width = width
        self.height = height
        self.me = me
        self.q = q
        self.remember_me = me
        self.gold = gold
        self.vampires = vampires
        self.obstacles = obstacles
        self.world = np.zeros((width, height), dtype=np.int8)
        x, y = me
        self.world[x, y] = self.ME
        x, y = gold
        self.world[x, y] = self.GOLD
        for point in obstacles:
            x, y = point
            self.world[x, y] = self.HOLE
        for point in vampires:
            x, y = point
            self.world[x, y] = self.VAMPIRE

    def __go_loading(self, i, total, show):
        percentage = i * 100 / total
        p = percentage
        s = "["
        k = 0
        while percentage > 0:
            s += "*"
            percentage -= 2
            k += 1
        k_prime = k
        while k < 50:
            k += 1
            s += " "
        if not (show == k_prime):
            # os.system('cls' if os.name == 'nt' else 'clear')
            print(
                s + "]" + str("{:2.3f}".format(p)).zfill(6) + "%  -->  " + str(i).zfill(5) + "/" + str(total).zfill(4))
        return k_prime

    def train(self, episodes=600, maximum_movements=200, random=0.98, discount=0.99999, gamma=0.99):
        show = -1
        lol = []
        epsilon = 1
        lamda = 1
        ans = []
        Q = np.multiply(np.ones((self.width, self.height, self.ACTION_NUMBERS)), self.q)
        for i in np.arange(episodes):
            lol.append(i+1)
            show = self.__go_loading(i, episodes, show)
            self.__init__(self.width, self.height, self.obstacles, self.remember_me, self.vampires, self.gold, self.q)
            for j in np.arange(maximum_movements):
                what = np.random.random(1)[0]
                x, y = self.me
                if what < epsilon:
                    phase = np.random.randint(0, self.ACTION_NUMBERS)
                else:
                    phase = np.argmax(Q[x, y])
                action = self.ACTIONS[phase]
                finish, reward = self.move(action)
                new_x, new_y = self.me
                Q[x, y, phase] = lamda * (reward + gamma * np.max(Q[new_x, new_y, phase])) + (1 - lamda) * Q[x, y, phase]
                if finish:
                    if reward == 100:
                        ans.append([{"episode": str(i)}, {"moves": str(j)}])
                    break
            lamda *= discount
            epsilon *= random
        print(ans)
        self.Q = Q
        return Q

    def test(self):
        ans = []
        self.__init__(self.width, self.height, self.obstacles, self.remember_me, self.vampires, self.gold, self.q)
        finish = False
        while True:
            x, y = self.me
            ans.append(self.me)
            if finish:
                break
            action = self.ACTIONS[np.argmax(self.Q[x, y])]
            finish, _ = self.move(action)
        return ans

    def move(self, action):
        dx, dy = action
        x, y = self.me
        reward = -1
        new_me_x = x+dx
        new_me_y = y+dy
        finish = False
        gold_x, gold_y = self.gold
        if x == self.width-1 or x == 0 or y == 0 or y == self.height-1:
            if x == 0 and dx == -1:
                reward = -2
                new_me_x = 0
            elif x == self.width-1 and dx == 1:
                reward = -2
                new_me_x = self.width-1
            if y == 0 and dy == -1:
                reward = -2
                new_me_y = 0
            elif y == self.height-1 and dy == 1:
                reward = -2
                new_me_y = self.height-1
        if new_me_x == gold_x and new_me_y == gold_y:
            finish = True
            reward = 100
        for vampire in self.vampires:
            vampire_x, vampire_y = vampire
            if vampire_x == new_me_x and vampire_y == new_me_y:
                finish = True
                reward = -100
        for hole in self.obstacles:
            hole_x, hole_y = hole
            if hole_x == new_me_x and hole_y == new_me_y:
                finish = True
                reward = -100
        if not finish:
            self.world[x, y] = self.EMPTY
            self.world[new_me_x, new_me_y] = self.ME
            self.me = (new_me_x, new_me_y)
        elif finish and reward == 100:
            self.world[x, y] = self.EMPTY
            self.world[new_me_x, new_me_y] = self.ME
            self.me = (new_me_x, new_me_y)
        return finish, reward

    def __str__(self):
        return self.world.__str__()+"\n********************\n********************\n"


def background(weight, obstacles, vampires, gold):
    length = weight*100
    win = GraphWin('My WORLD', length, length)
    for i in range(weight+1):
        l = Line(Point(100 * i, length), Point(100 * i, 0))
        l.setWidth(5)
        l.draw(win)
    for i in range(weight+1):
        l = Line(Point(0, 100 * i), Point(length, 100 * i))
        l.setWidth(5)
        l.draw(win)
    for obstacle in obstacles:
        x, y = obstacle
        obst = Rectangle(Point(y*100, x*100), Point((y+1)*100, (x+1)*100))
        obst.setFill("black")
        obst.draw(win)
    x, y = gold
    back = Rectangle(Point(y * 100, x * 100), Point((y + 1) * 100, (x + 1) * 100))
    back.setFill("cyan")
    back.draw(win)
    goal = Rectangle(Point(y*100+25, x*100+25), Point((y+1)*100-25, (x+1)*100-25))
    goal.setFill("brown")
    circle = Circle(Point(y*100+50, x*100+25), 25)
    circle.setFill("gold")
    circle.draw(win)
    goal.draw(win)
    line = Rectangle(Point(y*100+30, x*100+40), Point(y*100+70, x*100+45))
    line.setFill("black")
    lined = Rectangle(Point(y * 100 + 30, x * 100 + 30), Point((y + 1) * 100 - 30, (x + 1) * 100 - 30))
    lined.setFill("pink")
    lined.draw(win)
    lind = Rectangle(Point(y * 100 + 30, x * 100 + 55), Point(y * 100 + 70, x * 100 + 60))
    lind.setFill("black")
    lind.draw(win)
    line.draw(win)

    for vampire in vampires:
        x, y = vampire

        back = Rectangle(Point(y * 100, x * 100), Point((y + 1) * 100, (x + 1) * 100))
        back.setFill("purple")
        back.draw(win)

        head = Circle(Point(((weight-1)*100+50)-(weight-1-y)*100, ((weight-1)*100+50)-(weight-1-x)*100), 25)
        head.setFill("yellow")
        head.draw(win)

        eye1 = Circle(Point(((weight-1)*100+40)-(weight-1-y)*100, ((weight-1)*100+45)-(weight-1-x)*100), 5)
        eye1.setFill('red')
        eye1.draw(win)

        eye2 = Circle(Point(((weight-1)*100+60)-(weight-1-y)*100, ((weight-1)*100+45)-(weight-1-x)*100), 5)
        eye2.setFill('red')
        eye2.draw(win)

        mouth = Oval(Point(((weight-1)*100+40)-(weight-1-y)*100, ((weight-1)*100+65)-(weight-1-x)*100), Point(((weight-1)*100+60)-(weight-1-y)*100, ((weight-1)*100+65)-(weight-1-x)*100))
        mouth.setFill("red")
        mouth.draw(win)

        teeth_1 = []
        v1 = Point(((weight-1)*100+40)-(weight-1-y)*100, ((weight-1)*100+65)-(weight-1-x)*100)
        v2 = Point(((weight-1)*100+40)-(weight-1-y)*100+10, ((weight-1)*100+65)-(weight-1-x)*100)
        v3 = Point(((weight-1)*100+40)-(weight-1-y)*100+5, ((weight-1)*100+65)-(weight-1-x)*100+5)
        teeth_1.append(v1)
        teeth_1.append(v2)
        teeth_1.append(v3)
        triangle = Polygon(teeth_1)
        triangle.setFill("white")
        triangle.draw(win)

        teeth_2 = []
        v1 = Point(((weight-1)*100+60)-(weight-1-y)*100, ((weight-1)*100+65)-(weight-1-x)*100)
        v2 = Point(((weight-1)*100+60)-(weight-1-y)*100-10, ((weight-1)*100+65)-(weight-1-x)*100)
        v3 = Point(((weight-1)*100+60)-(weight-1-y)*100-5, ((weight-1)*100+65)-(weight-1-x)*100+5)
        teeth_2.append(v1)
        teeth_2.append(v2)
        teeth_2.append(v3)
        triangle = Polygon(teeth_2)
        triangle.setFill("white")
        triangle.draw(win)
    return win


def i_got_the_move(win, shape_list, x, y, color, weight):
    for shape in shape_list:
        shape.undraw()
        head = Circle(Point(((weight-1)*100+50) - (weight - 1 - y) * 100, ((weight-1)*100+50) - (weight - 1 - x) * 100), 25)
    head.setFill(color[0])
    head.draw(win)

    eye1 = Circle(Point(((weight-1)*100+40) - (weight - 1 - y) * 100, ((weight-1)*100+45) - (weight - 1 - x) * 100), 5)
    eye1.setFill(color[1])
    eye1.draw(win)

    eye2 = Circle(Point(((weight-1)*100+60) - (weight - 1 - y) * 100, ((weight-1)*100+45) - (weight - 1 - x) * 100), 5)
    eye2.setFill(color[1])
    eye2.draw(win)

    mouth = Oval(Point(((weight-1)*100+40) - (weight - 1 - y) * 100, ((weight-1)*100+65) - (weight - 1 - x) * 100),
                 Point(((weight-1)*100+60) - (weight - 1 - y) * 100, ((weight-1)*100+65) - (weight - 1 - x) * 100))
    mouth.setFill(color[2])
    mouth.draw(win)

    shape_list = [head, eye1, eye2, mouth]
    return shape_list


def human_and_colors(win, weight, x, y):
    head = Circle(Point(((weight-1)*100+50) - (weight - 1 - y) * 100, ((weight-1)*100+50) - (weight - 1 - x) * 100), 25)
    head.setFill("yellow")
    head.draw(win)

    eye1 = Circle(Point(((weight-1)*100+40) - (weight - 1 - y) * 100, ((weight-1)*100+45) - (weight - 1 - x) * 100), 5)
    eye1.setFill('blue')
    eye1.draw(win)

    eye2 = Circle(Point(((weight-1)*100+60) - (weight - 1 - y) * 100, ((weight-1)*100+45) - (weight - 1 - x) * 100), 5)
    eye2.setFill('blue')
    eye2.draw(win)

    mouth = Oval(Point(((weight-1)*100+40) - (weight - 1 - y) * 100, ((weight-1)*100+65) - (weight - 1 - x) * 100),
                 Point(((weight-1)*100+60) - (weight - 1 - y) * 100, ((weight-1)*100+65) - (weight - 1 - x) * 100))
    mouth.setFill("red")
    mouth.draw(win)

    human = []
    human.append(head)
    human.append(eye1)
    human.append(eye2)
    human.append(mouth)
    colors = []
    colors.append("yellow")
    colors.append("blue")
    colors.append("red")
    return human, colors


def run(width, height, obstacles, me, vampires, gold, episodes, q):
    game = Wampus(width, height, obstacles, me, vampires, gold, q)
    print(game)
    game.train(episodes=episodes)
    ans = game.test()
    print(ans)
    win = background(width, obstacles, vampires, gold)
    human, color = human_and_colors(win, height, me[0], me[1])
    time.sleep(1.5)
    for m in range(len(ans)):
        human = i_got_the_move(win, human, ans[m][0], ans[m][1], color, height)
        time.sleep(.75)
    win.close()
    win = GraphWin('You Solved the problem', width * 100, 200)
    win.setBackground("yellow")
    text = Text(Point(width * 100 / 2, 100), "The Gold has been found")
    text.setFill("red")
    text.draw(win)
    time.sleep(.8)
    time.sleep(.5)
    win.close()


def main():
    width = 5
    height = width
    obstacles = [(2, 3), (2, 2)]
    me = (4, 4)
    vampires = [(3, 1)]
    gold = (0, 0)
    run(width, height, obstacles, me, vampires, gold, 500, 0)
    run(width, height, obstacles, me, vampires, gold, 500, 1)
    run(width, height, obstacles, me, vampires, gold, 500, -1)

    width = 5
    height = width
    obstacles = [(2, 3), (2, 2)]
    me = (4, 4)
    vampires = [(3, 1)]
    gold = (2, 1)
    run(width, height, obstacles, me, vampires, gold, 500, 0)

    width = 5
    height = width
    obstacles = [(2, 3), (2, 2), (3, 4)]
    me = (4, 4)
    vampires = [(3, 1), (2, 1)]
    gold = (0, 0)
    run(width, height, obstacles, me, vampires, gold, 5000, 0)
    width = 4
    height = width
    obstacles = [(2, 3), (2, 2)]
    me = (3, 3)
    vampires = [(1, 1)]
    gold = (1, 2)
    run(width, height, obstacles, me, vampires, gold, 500, 0)

    width = 6
    height = width
    obstacles = [(2, 3), (2, 2)]
    me = (4, 4)
    vampires = [(1, 1), (3, 4)]
    gold = (1, 2)
    run(width, height, obstacles, me, vampires, gold, 5000, 0)


if __name__ == '__main__':
    main()
