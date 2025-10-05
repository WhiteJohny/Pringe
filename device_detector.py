from torch import accelerator, device

DEVICE = accelerator.current_accelerator().type if accelerator.is_available() else device("cpu")

print(f"Using {DEVICE} device")
