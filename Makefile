DATASETS_DIR = moot/optimize
COMMAND_FILE = commands.sh
NAME ?= SMAC
BUDGETS ?= 50 100 200

BASE_CMD = python3 experiment_runner_cluster.py \
	--name $(NAME) \
	--repeats 10 \
	--runs_output_folder ../results/results_$(NAME) \
	--logging_folder ../logging/logging_$(NAME) \
	--output_directory ../results/tmp/$(NAME)_tmp

generate-commands:
	@echo "#!/bin/bash" > $(COMMAND_FILE)
	@find $(DATASETS_DIR) -type f -name "*.csv" | while read dataset; do \
		for B in $(BUDGETS); do \
			echo "$(BASE_CMD) --datasets ../$$dataset --budget $${B};" >> $(COMMAND_FILE); \
		done \
	done
	@echo "wait" >> $(COMMAND_FILE)
	@chmod +x $(COMMAND_FILE)
	@mv $(COMMAND_FILE) experiments/$(COMMAND_FILE)

run-commands:
	@cd experiments && nohup ./$(COMMAND_FILE) > run.log 2>&1 &
	@echo "Commands are running in the background. Output is in experiments/run.log"

convert-commands:
	@mkdir -p jobs_$(NAME)
	@while read -r line; do \
		clean_line=$$(echo "$$line" | sed 's/;//'); \
		dataset=$$(echo "$$clean_line" | sed -n 's/.*--datasets \([^ ]*\.csv\).*/\1/p'); \
		base=$$(basename "$$dataset" .csv); \
		for B in $(BUDGETS); do \
			jobfile=jobs_$(NAME)/job_$${base}_$${B}.lsf; \
			echo "#!/bin/bash -l" > $$jobfile; \
			echo "#BSUB -J $(NAME)_$${base}_$${B}" >> $$jobfile; \
			echo "#BSUB -n 1" >> $$jobfile; \
			echo "#BSUB -q short" >> $$jobfile; \
			echo "#BSUB -W 120" >> $$jobfile; \
			echo "#BSUB -o out_%J.log" >> $$jobfile; \
			echo "#BSUB -e err_%J.log" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "source /usr/local/apps/miniconda20240526/etc/profile.d/conda.sh" >> $$jobfile; \
			echo "conda activate /share/path/path/ParetoMyth/env" >> $$jobfile; \
			echo "" >> $$jobfile; \
			echo "cd /share/path/path/ParetoMyth/experiments" >> $$jobfile; \
			echo "$$clean_line --budget $${B}" >> $$jobfile; \
		done; \
	done < experiments/$(COMMAND_FILE)

submit-jobs:
	@for f in jobs_$(NAME)/*.lsf; do \
		echo "bsub < $$f"; \
		bsub < $$f; \
	done
