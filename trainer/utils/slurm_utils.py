from collections import Counter
from time import sleep
from tqdm import tqdm


def track_jobs_with_pbar(jobs):
    num_completed = 0
    with tqdm(total=len(jobs)) as pbar:
        while any(job.state not in ["COMPLETED", "FAILED", "DONE"] for job in jobs):
            sleep(2)
            job_infos = [j.get_info() for j in jobs]
            state2count = Counter([info['State'] if 'State' in info else "None" for info in job_infos])
            newly_completed = state2count["COMPLETED"] - num_completed
            pbar.update(newly_completed)
            num_completed = state2count["COMPLETED"]
            s = [f"{k}: {v}" for k, v in state2count.items()]
            pbar.set_description(" | ".join(s))
    return num_completed
