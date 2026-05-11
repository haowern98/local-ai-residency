#include "reload/mmap_file.h"

#include <windows.h>
#include <memoryapi.h>

#include <cstdint>
#include <string>

namespace mosaicvram {

namespace {

void* OpenFileHandle(const std::string& path, std::size_t* file_size) {
  const std::wstring wide_path(path.begin(), path.end());

  HANDLE handle = CreateFileW(
      wide_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
      OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return nullptr;
  }

  LARGE_INTEGER size;
  if (GetFileSizeEx(handle, &size) == 0) {
    CloseHandle(handle);
    return nullptr;
  }
  *file_size = static_cast<std::size_t>(size.QuadPart);
  if (*file_size == 0) {
    CloseHandle(handle);
    return nullptr;
  }
  return handle;
}

void* CreateMappingHandle(void* file_handle) {
  HANDLE mapping = CreateFileMappingW(static_cast<HANDLE>(file_handle), nullptr,
                                      PAGE_READONLY, 0, 0, nullptr);
  return mapping;
}

void* MapViewOfMapping(void* mapping_handle) {
  return MapViewOfFile(static_cast<HANDLE>(mapping_handle), FILE_MAP_READ, 0, 0,
                       0);
}

bool PrefetchPageRange(void* address, std::size_t size) {
  WIN32_MEMORY_RANGE_ENTRY entry;
  entry.VirtualAddress = address;
  entry.NumberOfBytes = size;
  return PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0) != 0;
}

}  // namespace

MmapFile::~MmapFile() { Close(); }

MmapFile::MmapFile(MmapFile&& other) noexcept { MoveFrom(&other); }

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
  if (this != &other) {
    Close();
    MoveFrom(&other);
  }
  return *this;
}

bool MmapFile::Open(const std::string& path) {
  Close();

  void* handle = OpenFileHandle(path, &file_size_);
  if (handle == nullptr) {
    return false;
  }
  file_handle_ = handle;

  void* mapping = CreateMappingHandle(file_handle_);
  if (mapping == nullptr) {
    CloseHandle(static_cast<HANDLE>(file_handle_));
    file_handle_ = nullptr;
    file_size_ = 0;
    return false;
  }
  mapping_handle_ = mapping;

  view_ = MapViewOfMapping(mapping_handle_);
  if (view_ == nullptr) {
    CloseHandle(static_cast<HANDLE>(mapping_handle_));
    mapping_handle_ = nullptr;
    CloseHandle(static_cast<HANDLE>(file_handle_));
    file_handle_ = nullptr;
    file_size_ = 0;
    return false;
  }

  return true;
}

void MmapFile::Close() {
  if (view_ != nullptr) {
    UnmapViewOfFile(view_);
    view_ = nullptr;
  }
  if (mapping_handle_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(mapping_handle_));
    mapping_handle_ = nullptr;
  }
  if (file_handle_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(file_handle_));
    file_handle_ = nullptr;
  }
  file_size_ = 0;
}

bool MmapFile::PrefetchChunk(std::size_t offset, std::size_t size) {
  if (view_ == nullptr) {
    return false;
  }
  if (offset >= file_size_) {
    return false;
  }
  const std::size_t clamped_size =
      offset + size > file_size_ ? file_size_ - offset : size;
  if (clamped_size == 0) {
    return true;
  }

  void* address = static_cast<std::uint8_t*>(view_) + offset;
  return PrefetchPageRange(address, clamped_size);
}

void MmapFile::MoveFrom(MmapFile* other) {
  file_handle_ = other->file_handle_;
  mapping_handle_ = other->mapping_handle_;
  view_ = other->view_;
  file_size_ = other->file_size_;
  other->file_handle_ = nullptr;
  other->mapping_handle_ = nullptr;
  other->view_ = nullptr;
  other->file_size_ = 0;
}

}  // namespace mosaicvram
