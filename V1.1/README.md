Version 1.1 is the same as the parent directory versions only the network save file is smaller as it no longer contains output, pooling, 'momentum', and error layers. This comes at a minor cost for anyone using the `LINUX_DEBUG` feature as these layers can no longer be exported. But, this does make for a better release.