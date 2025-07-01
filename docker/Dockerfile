FROM python3.12

WORKDIR /mnt/container/tmp

RUN apt-get update && apt-get install -y \
r-base \
libharfbuzz-dev \
libfribidi-dev \
libfreetype6-dev \ 
libpng-dev \
libtiff5-dev \
libjpeg-dev \
cmake \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN R -q -e 'install.packages(c("tidyverse", "lme4"))'
RUN R -q -e 'install.packages("vscDebugger", repos = "https://manuelhentschel.r-universe.dev")'

CMD bash
