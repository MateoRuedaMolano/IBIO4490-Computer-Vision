LABORATORIO 1 Y 2
Computer Vision
Universidad de los Andes
Mateo Santiago Rueda Molano - 201517252

1) El comando grep se utiliza para buscar un patrón de texto (especificado en la línea de comando) en un archivo, múltiples archivos
o en una entrada. Se busca línea por línea el patrón y grep entrega la línea de texto completa donde se encuentra este patrón.

Sintaxis.
grep [OPCIONES] PATRÓN [ARCHIVO]

Esta información se basó de https://www.computerhope.com/unix/ugrep.htm

2) #!/bin/python corresponde a un shebang, se conoce como shebang al conjunto de caracteres #! cuando están al principio de un archivo 
tipo texto. Este indica que el archivo corresponde a un script y qué intérprete ha de usarse para ejecutar el mismo (en este caso el
intérprete es Python localizado en /bin).  Los sistemas operativos Linux (y otros sistemas Unix) soportan de forma nativa 
esta característica.

La información se basó del siguiente link https://bash.cyberciti.biz/guide/Shebang.

3)  
COMANDOS:
El comando utilizado para descargar la base de datos fue:
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
Al descomprimir se utilizó:
tar -xvf BSR_bsds500.tgz

4) 
COMANDOS:
vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ du BSR
vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ du -h BSR 

Se utilizó el comando du en la carpeta descomprimida y se obtuvo que esta ocupa un espacio de 73Megabytes o 74128bits. 
El -h se utilizó para saber el espacio que ocupaba en derivados del Byte (KiloByte, MegaByte...):

Como todas las imágenes son jpg, se determinó el número de imágenes mediante la línea de código

vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ find . -name "*.jpg" -exec identify {} \; | wc -l


Cualquier archivo con el nombre jpg, mediante un contador se almacena. El resultado fue de 500 imágenes.
Vale la pena aclarar que esta línea de código se corrió en la carpeta images dentro del dataset.
 
5) 

FORMATO DE LOS ARCHIVOS DE LA CARPETA IMAGES:

vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ find ./* -type f | awk -F. '{print $NF}' | sort -u

Resultado:
db
jpg
 
Mediante el comando anterior, se encontró la última línea después de un punto, que corresponde al formato de los archivos. 
Después de obtener los formatos de cada imagen, se ordenó con sort -u para obtener los tipos de formatos de las imágenes. 
Las imágenes así, son jpg, aunque existe un archivo en el folder 
de formato db. 

El separador de punto en awk y el print de la última línea se basó en los ejemplos de los siguientes links.
https://www.computerhope.com/unix/usort.htm y https://stackoverflow.com/questions/4304917/how-to-print-last-two-columns-using awk.


RESOLUCIÓN:

vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ identify $(find . -name "*.jpg") | awk '{print $3}' | sort -u

Resultado: 
321x481
481x321

Este comando se utilizó para hallar las resoluciones de las imágenes. Mediante $(find . -name "*.jpg") se encontraban los nombres de las imágenes
y haciendo identify se obtienen elementos como el formato, la resolución, los bits, si es RGB, etc. A partir de estos datos se buscó mediante awk 
la columna 3 correspondiente a la resolución. Con sort -u se unificaron las resoluciones iguales obteniendo tan solo las resoluciones 481*321 
y 321*481.

FORMATO TAN SOLO DE LAS IMÁGENES:

vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ identify $(find . -name "*.jpg") | awk '{print $2}' | sort -u

Resultado:
JPEG

Este comando es similar al anterior, solo que se obtiene en vez de las resoluciones, el formato de las imágenes.

La información de identify en imagemagick se obtuvo de:
https://superuser.com/questions/275502/how-to-get-information-about-an-image-picture-from-the-linux-command-line.

6) 


COMANDO ORIENTACIÓN LANDSCAPE:
vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ identify $(find . -name "*.jpg") | awk '{print $3}' | grep -c "481x321"

Resultado:
348

COMANDO ORIENTACIÓN PORTRAIT:
vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ identify $(find . -name "*.jpg") | awk '{print $3}' | grep -c "321x481"

Resultado:
152


El comando se basó en el del numeral anterior en el que obteníamos la resolución de cada imagen, sin embargo, se añadió una línea de grep -c "resolución",
la cual me buscaba en cada una de las resoluciones de las imágenes alguna de las dos encontradas en el numeral anterior. Mediante un contador, se obtiene
finalmente el resultado de todas las imágenes con las resoluciones tipo landscape (en este caso 481x321) y portrait (321x481).



7) Para el punto 7, primero se comenzó creando los directorios correspondientes de la nueva carpeta

COMANDOS:
mkdir newFolder 
cd newFolder
mkdir train
mkdir test
mkdir val

En la carpeta newFolder se crearon estos sub directorios para guardar las imágenes modificadas en resolución.

Posteriormente, se cambió el size de cada imagen de cada una de las carpetas (train, val y test) a 256x256 y se pasó a las carpetas nuevas creadas. Se usaron
3 comandos para cada una de las carpetas, los cuales fueron:
TRAIN:
vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ for imagen in $(find train -type f -name "*.jpg"); do convert $imagen -resize 256x256\! newFolder/$imagen; done

VAL:
vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ for imagen in $(find val -type f -name "*.jpg"); do convert $imagen -resize 256x256\! newFolder/$imagen; done

TEST:
vision@bcv002:~/ms.rueda10/BSR/BSDS500/data/images$ for imagen in $(find test -type f -name "*.jpg"); do convert $imagen -resize 256x256\! newFolder/$imagen; done

Así pues, las imágenes quedaron con resolución 256x256 y con el mismo nombre que las imágenes originales. Quedaron almacenadas en newFolder en cada una de las sub
carpetas asociadas (train,val, test).

El for, se basó en la sintaxis presentada en la guía del laboratorio, mientras que el resize se encontró de https://imagemagick.org/Usage/resize/






