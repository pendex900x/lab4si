import matplotlib.pyplot as plt
import numpy as np

buenosconsejos=0.9

class CliffWalking():
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
        self.agentPos = [0, 0]
        # acciones
        self.arriba = 0
        self.abajo = 1
        self.derecha = 2
        self.izquierda = 3
        self.acciones = [self.arriba, self.abajo,
                         self.derecha, self.izquierda]

        # zonas
        self.startPos = [0, 3]
        self.goalPos = [11, 3]
    # end __init__

    def reset(self):
        self.agentPos = self.startPos
        return self.agentPos
    # end reset

    def actuar(self, accion):
        x, y = self.agentPos

        if(accion == self.arriba):
            y = y -1
            if(y<0):
                y = 0
        elif(accion == self.abajo):
            y = y +1
            if(y >= self.alto):
                y = self.alto -1
        elif(accion == self.derecha):
            x = x +1
            if(x >= self.ancho):
                x = self.ancho -1
        elif(accion == self.izquierda):
            x = x -1
            if(x<0):
                x = 0
        else:
            print('Accion desconocida')

        estado = [x, y]
        reward = -1
        # x [1;10]
        # y = 3
        # cliff
        if(accion == self.abajo and y == 2
           and 1 <= x <= 10) or (
            accion == self.derecha
            and self.agentPos == self.startPos): #empieza precipicio

            reward = -200
            estado = self.startPos

        self.agentPos = estado

        return self.agentPos, reward
    # end actuar

class AgenteQLearning():
    def __init__(self, entorno, alpha = 0.5, epsilon = 0.1, gamma = 1):#0.99):
        self.entorno = entorno
        self.nEstados = [entorno.ancho, entorno.alto]
        self.nAcciones = 4

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros([self.nEstados[0], self.nEstados[1], self.nAcciones])

        # print(self.Q)
    # end __init__
    #policy Epsilon-Greedy

class AgenteSarsa():
    def __init__(self, entorno, alpha = 0.5, epsilon = 0.5, gamma = 1):#0.99):
        self.entorno = entorno
        self.nEstados = [entorno.ancho, entorno.alto]
        self.nAcciones = 4

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros([self.nEstados[0], self.nEstados[1], self.nAcciones])

        # print(self.Q)
    # end __init__


    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return np.random.randint(self.nAcciones)
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado[0], estado[1], :])
    # end seleccionarAccion
    def seleccionarAccionFeedBack(self,estado, entrenador, feedbackProbabilidad):

        if np.random.rand() <= feedbackProbabilidad:
            if np.random.rand()<=buenosconsejos:
                return np.argmax(entrenador.Q[estado[0], estado[1], :])

            else:
                return np.argmin(entrenador.Q[estado[0], estado[1], :])

        else: #accion agente
            return self.seleccionarAccion(estado)


    # td control
    def Sarsa(self, estado,accion,reward, estado_sig,accion_sig):
        td_target = reward + self.gamma * (self.Q[estado_sig[0], estado_sig[1], accion_sig]) #por la posicion x,y es que tiene 2 estados siguientes.
        td_error = td_target - self.Q[estado[0], estado[1], accion]
        self.Q[estado[0], estado[1], accion] += self.alpha * td_error


    def entrenar(self, episodios, entrenador=None,feed=0):
        recompensas = []

        for e in range(episodios):
            estado= self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                accion = self.seleccionarAccionFeedBack(estado,entrenador,feed)
                estado_sig, reward = self.entorno.actuar(accion)
                accion_sig = self.seleccionarAccion(estado_sig)

                recompensa += reward

                fin = self.entorno.goalPos == estado

                if not fin:
                    #actualiza valor Q
                    self.Sarsa(estado,accion,reward, estado_sig,accion_sig)
                estado = estado_sig

            recompensas.append(recompensa)

        return recompensas
    # end entrenar



cantidadAgentes = 50
episodios=500
entorno = CliffWalking(12, 4)
entrenador = AgenteSarsa(entorno)
qlearning = entrenador.entrenar(episodios)

aprendiz = AgenteSarsa(entorno)
ap=aprendiz.entrenar(episodios)

feedback = 0.5

rewardEntrenador=np.zeros(episodios)
rewardAprendiz=np.zeros(episodios)
for r in range(cantidadAgentes):
    print('Entrenando Agente autonomo: ',r)
    entrenador = AgenteSarsa(entorno)
    rewardEntrenador += entrenador.entrenar(episodios)


for r in range(cantidadAgentes):
    print('Entrenando Agente interactivo: ',r)
    aprendiz = AgenteSarsa(entorno)
    rewardAprendiz += aprendiz.entrenar(episodios,entrenador, feedback)


rewardAprendiz /= cantidadAgentes
rewardEntrenador /= cantidadAgentes

plt.plot(rewardAprendiz, label='Aprendiz')
plt.plot(rewardEntrenador, label='Entrenador')

plt.xlabel('Episodios')
plt.ylabel('Recompensa Promedio')
plt.ylim([-1000, -10])

plt.legend()
