"""Article 데이터베이스 모델 (DKaffeine-BE의 Article 엔티티와 매핑)"""
from datetime import datetime, date
from sqlalchemy import Column, BigInteger, Text, String, DateTime, Index
from sqlalchemy.sql import func

from app.infrastructure.persistence.models.base import Base


class Article(Base):
    """Article 엔티티 (Java 백엔드: com.dkaffein.news.domain.entity.Article와 일치)
    
    Java 백엔드 기준:
    - id: Long (PK, 컬럼명: id, @GeneratedValue)
    - publishedAt: LocalDateTime (컬럼명: published_at)
    - url: String (length=1000, NOT NULL, UNIQUE)
    - title: String (length=1000, NOT NULL)
    - description: String (TEXT)
    - imageUrl: String (length=1000, 컬럼명: image_url)
    - source: String (length=200)
    - BaseEntity 상속: createdAt, updatedAt, deletedAt
    """
    __tablename__ = "articles"

    # Java: @Id @GeneratedValue, 컬럼명: id
    id = Column("id", BigInteger, primary_key=True, autoincrement=True)
    
    # Java: publishedAt (LocalDateTime, 컬럼명: published_at)
    published_at = Column("published_at", DateTime, nullable=False)
    
    # Java: url (String, length=1000, NOT NULL, UNIQUE)
    url = Column("url", String(1000), nullable=False, unique=True)
    
    # Java: title (String, length=1000, NOT NULL)
    title = Column("title", String(1000), nullable=False)
    
    # Java: description (String, TEXT)
    description = Column("description", Text, nullable=True)
    
    # Java: imageUrl (String, length=1000, 컬럼명: image_url)
    image_url = Column("image_url", String(1000), nullable=True)
    
    # Java: source (String, length=200)
    source = Column("source", String(200), nullable=True)
    
    # Java BaseEntity: createdAt, updatedAt, deletedAt
    created_at = Column("created_at", DateTime, nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column("deleted_at", DateTime, nullable=True)

    def __repr__(self):
        return f"<Article(id={self.id}, title={self.title[:50]}..., published_at={self.published_at})>"


# 인덱스 생성 (published_at 기준 조회 최적화)
Index('ix_articles_published_at', Article.published_at)

