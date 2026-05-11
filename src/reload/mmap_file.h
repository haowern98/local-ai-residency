#ifndef MOSAICVRAM_SRC_RELOAD_MMAP_FILE_H_
#define MOSAICVRAM_SRC_RELOAD_MMAP_FILE_H_

#include <cstddef>
#include <string>

namespace mosaicvram {

class MmapFile {
 public:
  MmapFile() = default;
  ~MmapFile();

  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;
  MmapFile(MmapFile&& other) noexcept;
  MmapFile& operator=(MmapFile&& other) noexcept;

  bool Open(const std::string& path);
  void Close();
  bool PrefetchChunk(std::size_t offset, std::size_t size);

  const void* data() const { return view_; }
  std::size_t size() const { return file_size_; }
  bool is_open() const { return view_ != nullptr; }

 private:
  void MoveFrom(MmapFile* other);

  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
  void* view_ = nullptr;
  std::size_t file_size_ = 0;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RELOAD_MMAP_FILE_H_
