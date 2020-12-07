# pr3-atari-ai
Proyecto #3 y entrega final de la materia Inteligencia Artificial. Agente de Inteligencia Artificial que sea capaz de jugar un juego de Atari utilizando OpenAI Gym como entorno.

Para este proyecto se utilizió el repositorio de mrahtz (https://github.com/mrahtz/tensorflow-rl-pong) que está basado en la entrada del blog http://karpathy.github.io/2016/05/31/rl/ de Andrej Karpathy.
Se entrenó la inteligencia artificial desde 0 para mejor comprensión del funcionamiento del aprendizaje reforzado.

A palabras de Karpathy, el Pong es un excelente ejemplo de una simple tarea de aprendizaje reforzado. La inteligencia artificial controlara una de las barritas (la otra es controlada por otra AI decente, la del juego) y lo que se tiene que hacer es rebotar la pelota y que pase al otro jugador.

El juego básicamente funciona así: se recibe el frame de una imagen (un array de 210x160x3 byte (enteros que van de 0 a 255 dando valores de pixeles)). Y se debe decidir si la barrita debe ir hacia arriba o abajo (Que sería una decisión binaria). Después de cada elección que el juego ejecute la acción, dará una recompensa que será un +1 si la pelota pasó al otro jugador o un -1 si la pelota nos pasó a nosotros. El objetivo es que la barrita se mueva para obtener muchas recompensas.


### Uso
- Crear y activar virtualenv
- Se incluye requeriment.txt con las instalaciones necesarias
(practicamente: Python 3.8, OpenAI Gym y Tensorflow 2.2)

El programa se inicia escribiendo --load_checkpoint --render para ver como juega tras haber sido entrenada con ~5,000 episodios.
Dicho render se visualizará en tiempo con este tiempo 1/60

Si se quiere entrenar desde 0 no debe incluirse ningún argumento.

## Aprendizaje forzado - Introducción

Inicialmente en el aprendizaje reforzado (a diferencia del aprendizaje supervisado) se desconoce qué está bien o qué está mal de nuestra información, la única forma de saber esto al inicio es probando y ver si funciona. Se auto generan variables a traves de observaciones y retroalimentación.

Uno de los ejemplos que utiliza Soroush (thinkingparticle) para comprender mejor lo que es el aprendizaje forzado y sus implicaciones, nos pide que nos imaginemos un mundo imaginario donde somos gerentes de un nuevo departamento y no conocemos a nuestros empleados, no tenemos acceso a sus curriculum ni registros. No sabemos cual empleado es un diseñador, un programador, o del área de atención al cliente, etc.

Además de esto consideramos que no podemos comunicarnos con los empleados, solo podemos asignarles tareas a través del panel de una computadora y ver que tan rápido finalizan su asignación o verlos fallar.

El acercamiento a la solución de esta situación que propone es intentar y fallar, registrando la experiencia resultante. Entonces debemos ver los resultados y suponer los roles de los empleados. Y si actuamos conforme a esto y acertamos en nuestra supocision, entones repetimos el proceso hasta que conozcamos los roles de todos.

Agente: es aquel que está aprendiendo a ganar y adquiere beneficios para sí mismo (recompensas)
Ambiente(environment): El mundo con el que el agente puede interactuar. El agente "ve" states y hace "acciones" de acuerdo a lo que ve.
Recompensa(reward): El agente obtiene recompensas o castigos según sus acciones. 
Policy gradient: Una de las clases de algoritmos de control más importantes. Es un acercamiento para resolver problemas de aprendizaje reforzado.

# Network policy

Entre las primeras cosas que se planean es la política de red que se implementará a nuestro jugador (o agente en este caso). Esta red tomará el estatus del juego y decidirá que se debería hacer (moverse hacia arriba o abajo). Se usará un simple bloque a computar con 2-layer de redes neuronales que tomará una imagen raw en pixeles que producirá un simple número indicando la probabilidad de ir hacia arriba. Se distingue que en este caso es estándar utilizar una política estocástica, esto quiere decir que solo se producirá la probabilidad de que la AI mueva la barrita hacia arrba. En cada iteración se tomará una muestra de esta distribución, como tirar de una moneda, para obtener el movimiento que ocurrirá.

Se supone tendremos un vector que tendra información preprocesada de los pixeles.
Se inicializan dos matrices de manera random cuyo resultado se redondea en el rango de 0 y 1 (el primer valor esta dentro de la hidden layer). Una de las hidden layer detectará varios escenarios del juego y el segundo valor decidirá cada caso si debería ir hacia arriba o abajo.

Estan recomendados alimentar la política de red con al menos 2 frames para esta red política para que pueda detectar movimiento.

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

La política de red se inicializa con ciertos parametros y juega 100 juegos de Pong (llamado despliegue de política) Donde se asume que cada juego consiste de 200 frames asi que en total se hacen 20,000 decisiones por ir "arriba" o "abajo" y por cada uno de estos nosotros sabemos el parametro del gradiente, el cual nos dice que deberiamos cambiar los parametros si queremos motivar la decision en este estado en el futuro. Todo loq ue queda es etiquetar cada decision que s egha hecho como buena o mala. Por ejemplo supongamos que ganamos 12 juegos y perdemos 88. Si tomamos todo 200*12 = 2400 decisiones que hemos hecho para ganar los juegos y obtener una actualizacion positiva (llenando un +1.0 en el gradiente de la accion ejemplificada). Y tomamos las otras 200*88 = 17600 decisiones en los juegos que se perdieron y hacemos una actualizacion negativa (desmotivando lo que hicimos). Y asi el sistema se inclinara mas a repetir acciones que funcionaron y tendra menos probabilidades de repetir acciones que no funcionaron. Asi que ahora solo deben jugarse otros 100 juegos con este sistema y repetirse una y otra vez.

# 


## Fuentes y referencias
https://github.com/mrahtz/tensorflow-rl-pong
https://www.tensorflow.org/guide/upgrade
https://gym.openai.com/envs/#atari
https://github.com/thinkingparticle/deep_rl_pong_keras

