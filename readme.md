<h1> Reversi Tournament: </h1>

Es una plataforma que permite cargar modelos de RL como jugadores y hacerlos competir entre sí,
mediante la realización de un torneo.

<h3> Reglas del torneo: </h3>

* La modalidad es "todos contra todos". Cada jugador juega un partido contra cada uno del resto
de los competidores.
* El partido consiste en 1 o más juegos (games), por defecto 100, y quien gane la mayoría es el
ganador. Si ganan la misma cantidad de juegos (50/50), el partido se considera empatado.
* Cuando un jugador gana un partido suma 3 puntos, si empata 1 y si pierde 0.
* El o los ganadores/ganadoras del torneo son aquellos que sumen la mayor cantidad de puntos al finalizar
todos los partidos.

<h3> Instalación: </h3>

Instalar las siguientes dependencias:
* torch
* dill
* boardgame2

<h3> Instrucciones para agregar un jugador propio: </h3>

* Se debe agregar un módulo (archivo .py, por ejemplo jperez.py) con todas las clases custom necesarias para cargar el
modelo: features extractors, custom Actor-Critic policies, y la clase para el jugador,
descrito en el siguiente punto. El motivo de poner el nombre propio es el de asegurarse de que no haya conflictos entre
clases con el mismo nombre, pero de jugadores diferentes. <u>Importante:</u> El modelo tiene que haber sido entrenado utilizando este módulo para
cargarse luego correctamente.
* El único requisito para la clase del jugador (ej JPerezPlayer) es que herede de BaseTorchPlayer
(ver players.py), llamando correctamente al constructor de super e implementando el método
predict, abstracto en BasePlayer. Este es un paso necesario para que el torneo funcione correctamente.
* Agregar el archivo zip del modelo a cargar en la carpeta ./models/, con el nombre del
participante. Por ejemplo jperez.zip. El nombre del archivo se utiliza para identificar
unívocamente al jugador en el torneo.

<h3> Instrucciones para jugar el torneo: </h3>

* Una vez cargados los jugadores en la carpeta model, abrir la notebook "Torneo.ipynb" y seguir
los pasos descritos en la misma.

<h3> Opciones de la clase Tournament: </h3>

* games_per_match: Por cada partido entre jugadores, cuantos juegos se juegan para definir el ganador (3 puntos), que será quien más cantidad de juegos gane. Si empatan en cantidad de juegos ganados, entonces el partido termina en empate y cada jugador suma 1 punto.
* board_shape: Tamaño del tablero (n x n).
* deterministic: True si los jugadores deben jugar en modo determinístico, False en caso contrario.
* n_processes: Cantidad de procesos en paralelo a utilizar para jugar los partidos. Si n_processes = 1, no se crea ningún proceso nuevo (desactivado).
* verbose: Muestra información del torneo y resultados en la notebook (recomendado True).
* log_folder: Especifica la carpeta donde se generarán los logs del torneo. Si es None, no se creará ningún log file.
    * Cada torneo genera una carpeta nueva dentro de log_folder con el formato Tournament_fecha_hora. En ella se agregan:
        * Un archivo results.log con el/los ganadores/ganadoras del torneo y la tabla final de posiciones.
        * Un archivo match_jugador1_vs_jugador2.log por cada partido (combinación de jugadores), con las estadísticas de ese partido.