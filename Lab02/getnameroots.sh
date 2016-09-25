#!/bin/bash

ls ./SpectralData/* | rev | cut -d/ -f 1 | cut -d '_' -f2- | rev | sort -u
