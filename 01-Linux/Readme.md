

# Introduction to Linux

## Preparation

1. Boot from a usb stick (or live cd), we suggest to use  [Ubuntu gnome](http://ubuntugnome.org/) distribution, or another ubuntu derivative.

2. (Optional) Configure keyboard layout and software repository
   Go to the the *Activities* menu (top left corner, or *start* key):
      -  Go to settings, then keyboard. Set the layout for latin america
      -  Go to software and updates, and select the server for Colombia
3. (Optional) Instead of booting from a live Cd. Create a partition in your pc's hard drive and install the linux distribution of your choice, the installed Os should perform better than the live cd.

## Introduction to Linux

1. Linux Distributions

   Linux is free software, it allows to do all sort of things with it. The main component in linux is the kernel, which is the part of the operating system that interfaces with the hardware. Applications run on top of it. 
   Distributions pack together the kernel with several applications in order to provide a complete operating system. There are hundreds of linux distributions available. In
   this lab we will be using Ubuntu as it is one of the largest, better supported, and user friendly distributions.


2. The graphical interface

   Most linux distributions include a graphical interface. There are several of these available for any taste.
   (http://www.howtogeek.com/163154/linux-users-have-a-choice-8-linux-desktop-environments/).
   Most activities can be accomplished from the interface, but the terminal is where the real power lies.

### Playing around with the file system and the terminal
The file system through the terminal
   Like any other component of the Os, the file system can be accessed from the command line. Here are some basic commands to navigate through the file system

   -  ``ls``: List contents of current directory
   - ``pwd``: Get the path  of current directory
   - ``cd``: Change Directory
   - ``cat``: Print contents of a file (also useful to concatenate files)
   - ``mv``: Move a file
   - ``cp``: Copy a file
   - ``rm``: Remove a file
   - ``touch``: Create a file, or update its timestamp
   - ``echo``: Print something to standard output
   - ``nano``: Handy command line file editor
   - ``find``: Find files and perform actions on it
   - ``which``: Find the location of a binary
   - ``wget``: Download a resource (identified by its url) from internet 

Some special directories are:
   - ``.`` (dot) : The current directory
   -  ``..`` (two dots) : The parent of the current directory
   -  ``/`` (slash): The root of the file system
   -  ``~`` (tilde) :  Home directory
      

Using these commands, take some time to explore the ubuntu filesystem, get to know the location of your user directory, and its default contents. 

To get more information about a command call it with the ``--help`` flag, or call ``man <command>`` for a more detailed description of it, for example ``man find`` or just search in google.


## Input/Output Redirections
Programs can work together in the linux environment, we just have to properly 'link' their outputs and their expected inputs. Here are some simple examples:

1. Find the ```passwd```file, and redirect its contents error log to the 'Black Hole'
   >  ``find / -name passwd  2> /dev/null``

   The `` 2>`` operator redirects the error output to ``/dev/null``. This is a special file that acts as a sink, anything sent to it will disappear. Other useful I/O redirection operations are
      -  `` > `` : Redirect standard output to a file
      -  `` | `` : Redirect standard output to standard input of another program
      -  `` 2> ``: Redirect error output to a file
      -  `` < `` : Send contents of a file to standard input
      -  `` 2>&1``: Send error output to the same place as standard output

2. To modify the content display of a file we can use the following command. It sends the content of the file to the ``tr`` command, which can be configured to format columns to tabs.

   ```bash
   cat milonga.txt | tr '\n' ' '
   ```
   
## SSH - Server Connection

1. The ssh command lets us connect to a remote machine identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER (**vision** in our case). The second command allows us to copy files between systems (you will get the actual login information in class).

   ```bash
   
   #connect
   ssh USER@SERVER
   ```

2. The scp command allows us to copy files form a remote server identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER. Following the SERVER information, we add ':' and write the full path of the file we want to copy, finally we add the local path where the file will be copied (remember '.' is the current directory). If we want to copy a directory we add the -r option. for example:

   ```bash
   #copy 
   scp USER@SERVER:~/data/sipi_images .
   
   scp -r USER@SERVER:/data/sipi_images .
   ```
   
   Notice how the first command will fail without the -r option

See [here](ssh.md) for different types of SSH connection with respect to your OS.

## File Ownership and permissions   

   Use ``ls -l`` to see a detailed list of files, this includes permissions and ownership
   Permissions are displayed as 9 letters, for example the following line means that the directory (we know it is a directory because of the first *d*) *images*
   belongs to user *vision* and group *vision*. Its owner can read (r), write (w) and access it (x), users in the group can only read and access the directory, while other users can't do anything. For files the x means execute. 
   ```bash
   drwxr-x--- 2 vision vision 4096 ene 25 18:45 images
   ```

   -  ``chmod`` change access permissions of a file (you must have write access)
   -  ``chown`` change the owner of a file

## Sample Exercise: Image database

1. Create a folder with your Uniandes username. (If you don't have Linux in your personal computer)

2. Copy *sipi_images* folder to your personal folder. (If you don't have Linux in your personal computer)

3.  Decompress the images (use ``tar``, check the man) inside *sipi_images* folder. 

4.  Use  ``imagemagick`` to find all *grayscale* images. We first need to install the *imagemagick* package by typing

    ```bash
    sudo apt-get install imagemagick
    ```
    
    Sudo is a special command that lets us perform the next command as the system administrator
    (super user). In general it is not recommended to work as a super user, it should only be used 
    when it is necessary. This provides additional protection for the system.
    
    ```bash
    find . -name "*.tiff" -exec identify {} \; | grep -i gray | wc -l
    ```
    
3.  Create a script to copy all *color* images to a different folder
    Lines that start with # are comments
    
      ```bash
      #!/bin/bash
      
      # go to Home directory
      cd ~ # or just cd

      # remove the folder created by a previous run from the script
      rm -rf color_images

      # create output directory
      mkdir color_images

      # find all files whose name end in .tif
      images=$(find sipi_images -name *.tiff)
      
      #iterate over them
      for im in ${images[*]}
      do
         # check if the output from identify contains the word "gray"
         identify $im | grep -q -i gray
         
         # $? gives the exit code of the last command, in this case grep, it will be zero if a match was found
         if [ $? -eq 0 ]
         then
            echo $im is gray
         else
            echo $im is color
            cp $im color_images
         fi
      done
      
      ```
      -  save it for example as ``find_color_images.sh``
      -  make executable ``chmod u+x`` (This means add Execute permission for the user)
      -  run ``./find_duplicates.sh`` (The dot is necessary to run a program in the current directory)
    

## Your turn

1. What is the ``grep``command?

   El comando grep se utiliza para buscar un patrón de texto (especificado en la línea de comando) en un archivo, múltiples archivos o en una entrada. Se busca línea por línea el patrón y grep entrega la línea de texto completa donde se encuentra este patrón.

   ##### Sintaxis.

   ``grep`` [*opciones*] patrón [*archivo*]

   Esta información se basó de  [Computer Hope](https://www.computerhope.com/unix/ugrep.htm)

   

2. What is the meaning of ``#!/bin/python`` at the start of scripts?

   `#!/bin/python` corresponde a un shebang. Se conoce como shebang al conjunto de caracteres `#!` cuando están al principio de un archivo tipo texto. Este indica que el archivo corresponde a un script y qué intérprete ha de usarse para ejecutar el mismo (en este caso el intérprete es Python localizado en `/bin`).  Los sistemas operativos Linux (y otros sistemas Unix) soportan de forma nativa esta característica.

   

3. Download using ``wget`` the [*bsds500*](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) image segmentation database, and decompress it using ``tar`` (keep it in you hard drive, we will come back over this data in a few weeks).

   El comando utilizado para descargar la base de datos se muestra a continuación:

   ![](Imag_lab1/1.PNG)

   Al descomprimir se utilizó:

   ![](Imag_lab1/2.PNG)



4. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?

   ##### Tamaño *en* el disco de la base de datos descomprimida.

   Se utilizó el comando `du` en la carpeta descomprimida y se obtuvo que esta ocupa un espacio de 73Megabytes o 74128bits. El `-h` se utilizó para saber el espacio que ocupaba en derivados del Byte (KiloByte, MegaByte, etc.)

   Comandos y resultados:

  ![](Imag_lab1/3.PNG)

   Este comando se utiliza para determinar el tamaño en bits.  

   ![](Imag_lab1/4.PNG)

   Tamaño 74128 bits.

   ![](Imag_lab1/5.PNG)

   El comando anterior se utiliza para determinar el tamaño en derivados del Byte. 

   ![](Imag_lab1/6.PNG)

   Tamaño 73MBytes.

   La sintaxis y el uso de `du` en este problema, se obtuvo de [Ayuda Linux](https://ayudalinux.com/comandos-de-tamano-y-espacio-de-disco-df-y-du/)

   

   ##### Número de imágenes en el directorio

   Como todas las imágenes son *jpg* se determinó el número de imágenes mediante la línea de código:

   ![](Imag_lab1/7.PNG)

   Con un resultado como podemos evidenciar en la parte inferior de 500 imágenes. 

   Cualquier archivo con el nombre *jpg*, mediante una variable contador, se almacena en ella el resultado (se van contando así las imágenes). Vale la pena aclarar que esta línea de código se corrió en la carpeta images dentro del dataset

   

   

   5. What are all the different resolutions? What is their format? Tip: use ``awk``, ``sort``, ``uniq``

      ##### Formato

      ![](Imag_lab1/8.PNG)

   Mediante el comando anterior, se encontró la última línea después de un punto, que corresponde al formato de los archivos.  Después de obtener los formatos de cada imagen, se ordenó con `sort -u` para obtener los tipos de formatos de las imágenes.  Las imágenes así, son *jpg*, aunque existe un archivo en el folder de formato *db*. 

   

   El separador de punto en awk y el print de la última línea se basó en los ejemplos de los siguientes links.
   [Computer Hope](https://www.computerhope.com/unix/usort.htm) y [awk](https://stackoverflow.com/questions/4304917/how-to-print-last-two-columns-using-awk.)

   ##### Formato tan solo de las imágenes

   ![](Imag_lab1/9.PNG)

   Este comando es similar al anterior, solo que se obtiene en vez de las resoluciones, el formato de las imágenes.

   La información de identify en imagemagick se obtuvo de [identify](https://superuser.com/questions/275502/how-to-get-information-about-an-image-picture-from-the-linux-command-line.)

   

   ##### Resolución

   ![](Imag_lab1/10.PNG)
   
   Este comando se utilizó para hallar las resoluciones de las imágenes. Mediante `$(find . -name ".jpg")` se encontraban los nombres de las imágenes y haciendo `identify` se obtienen elementos como el formato, la resolución, los bits, si es RGB, etc. A partir de estos datos se buscó mediante `awk`  la columna 3 correspondiente a la resolución. Con `sort -u` se unificaron las resoluciones iguales obteniendo tan solo las resoluciones 481x321  y 321x481.

   

   6. How many of them are in *landscape* orientation (opposed to *portrait*)? Tip: use ``awk`` and ``cut``

     ![](Imag_lab1/11.PNG)

   El comando se basó en el del numeral anterior en el que obteníamos la resolución de cada imagen, sin embargo, se añadió una línea de `grep -c` "resolución", la cual me buscaba en cada una de las resoluciones de las imágenes alguna de las dos encontradas en el numeral anterior. Mediante un contador, se obtiene finalmente el resultado de todas las imágenes con las resoluciones tipo *landscape* (en este caso 481x321) y *portrait* (321x481).

7. Crop all images to make them square (256x256) and save them in a different folder. Tip: do not forget about  [imagemagick](http://www.imagemagick.org/script/index.php).

Para el punto 7, primero se comenzó creando los directorios correspondientes de la nueva carpeta

![](Imag_lab1/12.PNG)

En la carpeta newFolder se crearon estos sub directorios para guardar las imágenes modificadas en resolución.

Posteriormente, se cambió el size de cada imagen de cada una de las carpetas (*train, val y test*) a 256x256 y se pasó a las carpetas nuevas creadas. Se usaron 3 comandos para

 cada una de las carpetas, los cuales fueron:

##### Train:

![](Imag_lab1/13.PNG)

##### Validation:

![](Imag_lab1/14.PNG)

##### Test:

![](Imag_lab1/15.PNG)

Como vemos todas las imágenes quedaron en la nueva carpeta newFolder.

Ahora para verificar la resolución, sacamos para train la resolución con la línea de código del numeral 5.

![](Imag_lab1/16.PNG)

Nos paramos en la nueva carpeta (newFolder) y hacemos el mismo procedimiento del numeral 5. Nos damos cuenta que todas las imágenes tienen 256x256 por lo que está correcto el procedimiento. 

El for, se basó en la sintaxis presentada en la guía del laboratorio, mientras que el resize se encontró de [Imagemagick](https://imagemagick.org/Usage/resize/ ) 




