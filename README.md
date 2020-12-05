# pr3-atari-ai
Proyecto #3 y entrega final de la materia Inteligencia Artificial. Agente de Inteligencia Artificial que sea capaz de jugar un juego de Atari utilizando OpenAI Gym como entorno.

Para este proyecto se utilizió el repositorio de mrahtz (https://github.com/mrahtz/tensorflow-rl-pong) que está basado en la entrada del blog http://karpathy.github.io/2016/05/31/rl/ de Andrej Karpathy.
Se entrenó la inteligencia artificial desde 0 para mejor comprensión del funcionamiento del aprendizaje reforzado.

## Aprendizaje forzado - Introducción
Inicialmente en el aprendizaje reforzado (a diferencia del aprendizaje supervisado) se desconoce qué está bien o qué está mal de nuestra información, la única forma de saber esto al inicio es probando y ver si funciona. Se auto generan variables a traves de observaciones y retroalimentación.

Uno de los ejemplos que utiliza Soroush (thinkingparticle) para comprender mejor lo que es el aprendizaje forzado y sus implicaciones, nos pide que nos imaginemos un mundo imaginario donde somos gerentes de un nuevo departamento y no conocemos a nuestros empleados, no tenemos acceso a sus curriculum ni registros. No sabemos cual empleado es un diseñador, un programador, o del área de atención al cliente, etc.

Además de esto consideramos que no podemos comunicarnos con los empleados, solo podemos asignarles tareas a través del panel de una computadora y ver que tan rápido finalizan su asignación o verlos fallar.

El acercamiento a la solución de esta situación que propone es intentar y fallar, registrando la experiencia resultante. Entonces debemos ver los resultados y suponer los roles de los empleados. Y si actuamos conforme a esto y acertamos en nuestra supocision, entones repetimos el proceso hasta que conozcamos los roles de todos.

Agente: es aquel que está aprendiendo a ganar y adquiere beneficios para sí mismo (recompensas)
Ambiente(environment): El mundo con el que el agente puede interactuar. El agente "ve" states y hace "acciones" de acuerdo a lo que ve.
Recompensa(reward): El agente obtiene recompensas o castigos según sus acciones. 
Policy gradient: Una de las clases de algoritmos de control más importantes. Es un acercamiento para resolver problemas de aprendizaje reforzado.

## Policy Gradients 

En este algoritmo se usó una policy para ver que acciones llevan a recompensas altas para incrementar la probabilidad de que se repitan.
El proposito del aprendizaje forzado es encontrar la estrategia de comportamiento mas optima para que el agente reciba las mejores recompensas. 

# Stochastic policy 
Estocástico: Sometido al azar y que es objeto de analisis estadistico. Se aplica a procesos, algoritmos y modelos en los que existe una secuencia cambiante de eventos analizables probabilísticamente a medida que pasa el tiempo.
Toma ejemplos de acciones acciones y si esas acciones eventualmente ocurren y llevan a un buen desenlace son motivadas para que se repitan en el futuro, pero si esas acciones tomadas llevan aun mal resultando, son desmotivadas.



# Vocabulario:

    'Round': Un partida en la que un jugador gana un punto
    'Episode': una serie de rounds que determina un juego "completado" (Usualmente cerca 20 mas o menos segun la misma decisi'on del juego)

# Tiempo de entrenamiento

    Toma cerca de 500 episodios ver si el agente ha mejorado o no
    Toma cerca de 7000 episodios llegar a un estado en el que el agente gana la mitad y pierde la mitad de los round

# Protocolo de entrenamiento

La política 
Training protocol. So here is how the training will work in detail. We will initialize the policy network with some W1, W2 and play 100 games of Pong (we call these policy “rollouts”). Lets assume that each game is made up of 200 frames so in total we’ve made 20,000 decisions for going UP or DOWN and for each one of these we know the parameter gradient, which tells us how we should change the parameters if we wanted to encourage that decision in that state in the future. All that remains now is to label every decision we’ve made as good or bad. For example suppose we won 12 games and lost 88. We’ll take all 200*12 = 2400 decisions we made in the winning games and do a positive update (filling in a +1.0 in the gradient for the sampled action, doing backprop, and parameter update encouraging the actions we picked in all those states). And we’ll take the other 200*88 = 17600 decisions we made in the losing games and do a negative update (discouraging whatever we did). And… that’s it. The network will now become slightly more likely to repeat actions that worked, and slightly less likely to repeat actions that didn’t work. Now we play another 100 games with our new, slightly improved policy and rinse and repeat.

## Fuentes y referencias
https://github.com/mrahtz/tensorflow-rl-pong
https://www.tensorflow.org/guide/upgrade
https://gym.openai.com/envs/#atari
https://github.com/thinkingparticle/deep_rl_pong_keras

